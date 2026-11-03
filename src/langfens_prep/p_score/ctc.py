"""Independent log-space implementation of Cao et al., arXiv:2507.16838v3,
Eq. 4/12 and S/SD languages (§III-C). Graph: ctc-union-language-v1.

This is NOT author-code parity: no next-neighbor wildcard exclusion is used.
Occupancy is an exact conditional expectation via a forward reward semiring,
not the paper's normalized-forward approximation (Eq. 13). No autograd.
"""

from dataclasses import dataclass
import math
from typing import Sequence

NEG_INF = -math.inf
GRAPH_REVISION = "ctc-union-language-v1"


def logsumexp(values: Sequence[float]) -> float:
    maximum = max(values, default=NEG_INF)
    if maximum == NEG_INF:
        return NEG_INF
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in values))


def validate_emissions(
    emissions: Sequence[Sequence[float]], valid_frames: int | None = None,
    *, num_classes: int = 40,
) -> tuple[tuple[float, ...], ...]:
    """Validate log probabilities, then crop right padding; never renormalize.

    NaN/+inf are rejected even in padding. -inf denotes exact zero probability.
    Only valid rows must be normalized (absolute logsumexp tolerance 1e-5,
    suitable for an externally supplied FP32 log-softmax cache).
    """
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must include blank plus at least one phone")
    if valid_frames is None:
        valid_frames = len(emissions)
    if type(valid_frames) is not int or not 0 <= valid_frames <= len(emissions):
        raise ValueError("invalid valid_frames")
    result = []
    for t, row in enumerate(emissions):
        if len(row) != num_classes:
            raise ValueError("emission vocabulary/shape mismatch")
        if any(type(x) not in (float, int) or math.isnan(x) or x == math.inf for x in row):
            raise ValueError("emissions require numeric log probabilities without NaN/+inf")
        numeric = tuple(float(x) for x in row)
        if t < valid_frames:
            if any(x > 0 for x in numeric) or abs(logsumexp(numeric)) > 1e-5:
                raise ValueError("valid emission row is not normalized log probability")
            result.append(numeric)
    return tuple(result)


def validate_phones(phones: Sequence[int], num_classes: int) -> tuple[int, ...]:
    if any(type(p) is not int or not 1 <= p < num_classes for p in phones):
        raise ValueError("prompt phones must be nonblank vocabulary IDs")
    return tuple(phones)


def expanded_states(phones: Sequence[int]) -> tuple[int, ...]:
    states = [0]
    for phone in phones:
        states.extend((phone, 0))
    return tuple(states)


def predecessors(states: Sequence[int], state: int) -> tuple[int, ...]:
    result = [state]
    if state > 0:
        result.append(state - 1)
    if state > 1 and states[state] != 0 and states[state] != states[state - 2]:
        result.append(state - 2)
    return tuple(result)


@dataclass(frozen=True)
class ForwardResult:
    log_mass: float
    log_occupancy_mass: float

    @property
    def occupancy(self) -> float | None:
        if self.log_mass == NEG_INF:
            return None
        if self.log_occupancy_mass == NEG_INF:
            return 0.0
        return math.exp(self.log_occupancy_mass - self.log_mass)


def ctc_forward(
    log_emissions: Sequence[Sequence[float]], phones: Sequence[int],
    *, target_index: int | None = None,
) -> ForwardResult:
    """CTC forward over already validated arrays; empty target accepts all blanks.

    r(s) sums path_weight * number_of_target_state_visits. Updating r in log
    space gives exact posterior occupancy with only a forward pass. With no
    target_index all rewards are zero (used for the deletion alternative).
    """
    if target_index is not None and not 0 <= target_index < len(phones):
        raise ValueError("invalid target_index")
    states = expanded_states(phones)
    alpha = [NEG_INF] * len(states)
    reward = [NEG_INF] * len(states)
    alpha[0] = 0.0  # virtual time -1, before any emitted frames
    target_state = None if target_index is None else 2 * target_index + 1
    for row in log_emissions:
        next_alpha, next_reward = [], []
        for s, symbol in enumerate(states):
            incoming = predecessors(states, s)
            a = logsumexp([alpha[j] for j in incoming]) + row[symbol]
            r = logsumexp([reward[j] for j in incoming]) + row[symbol]
            if s == target_state:
                r = logsumexp([r, a])
            next_alpha.append(a)
            next_reward.append(r)
        alpha, reward = next_alpha, next_reward
    terminals = [len(states) - 1]
    if phones:
        terminals.append(len(states) - 2)
    return ForwardResult(logsumexp([alpha[j] for j in terminals]),
                         logsumexp([reward[j] for j in terminals]))


def alternatives(phones: Sequence[int], target_index: int, variant: str,
                 num_classes: int) -> tuple[tuple[int, ...], ...]:
    """Unique output strings; disjoint CTC preimages imply no path multiplicity.

    Deletion consumes no frame. Removing the target reconstructs the ordinary
    CTC graph, so newly adjacent equal neighbors STILL require a blank.
    """
    phones = validate_phones(phones, num_classes)
    if variant not in {"S", "SD"}:
        raise ValueError("only S and SD are supported; SDI is not registered")
    if type(target_index) is not int or not 0 <= target_index < len(phones):
        raise ValueError("invalid target_index")
    left, right = phones[:target_index], phones[target_index + 1:]
    sequences = {left + (q,) + right for q in range(1, num_classes)}
    if variant == "SD":
        sequences.add(left + right)
    return tuple(sorted(sequences))


@dataclass(frozen=True)
class PhoneScore:
    prompt_index: int
    variant: str
    canonical_log_mass: float
    relaxed_log_mass: float
    raw_gop: float | None
    s_error: float | None
    posterior_target_occupancy: float | None
    occupancy_normalized_error: float | None
    status: str
    reason: str | None


def score_prompt(
    emissions: Sequence[Sequence[float]], phones: Sequence[int], variant: str,
    *, valid_frames: int | None = None, num_classes: int = 40,
) -> list[PhoneScore]:
    """Error = log P(relaxed) - log P(canonical), not an error probability.

    All alternatives have unit weight; sum, never mean/max across strings.
    Invalid/nonfinite ratios abstain rather than becoming favorable infinities.
    """
    rows = validate_emissions(emissions, valid_frames, num_classes=num_classes)
    phones = validate_phones(phones, num_classes)
    if variant not in {"S", "SD"}:
        raise ValueError("only S and SD supported")
    canonical = ctc_forward(rows, phones).log_mass
    result = []
    for i in range(len(phones)):
        masses, rewards = [], []
        for seq in alternatives(phones, i, variant, num_classes):
            deletion = len(seq) < len(phones)
            forward = ctc_forward(rows, seq, target_index=None if deletion else i)
            masses.append(forward.log_mass)
            rewards.append(forward.log_occupancy_mass)
        denominator = logsumexp(masses)
        occupancy = ForwardResult(denominator, logsumexp(rewards)).occupancy
        reason = None
        if not rows:
            reason = "no_valid_frames"
        elif denominator == NEG_INF:
            reason = "relaxed_context_infeasible_or_zero_mass"
        elif canonical == NEG_INF:
            reason = "canonical_infeasible_or_zero_mass"
        error = None if reason else denominator - canonical
        if error is not None and not math.isfinite(error):
            error, reason = None, "nonfinite_log_ratio"
        if error is not None and error < -1e-10:
            raise ArithmeticError("canonical path set is not contained in relaxed set")
        result.append(PhoneScore(i, variant, canonical, denominator,
                                None if error is None else -error, error,
                                occupancy,
                                None if error is None or occupancy is None else
                                error / max(1.0, occupancy),
                                "abstain" if reason else "ok", reason))
    return result
