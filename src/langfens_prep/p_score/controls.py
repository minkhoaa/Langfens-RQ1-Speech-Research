"""Independent model-activation and greedy/edit controls; no gold inputs.

Recognition edits are NOT human operation labels. Canonical Viterbi forces
all prompt phones to occur and cannot locate a physically absent segment.
"""

from dataclasses import dataclass
import math
from typing import Sequence

from .ctc import NEG_INF, expanded_states, predecessors


def greedy_decode(rows: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Collapse consecutive frame repeats BEFORE blank removal; low-ID ties."""
    result, previous = [], None
    for row in rows:
        symbol = max(range(len(row)), key=row.__getitem__)
        if symbol != previous and symbol != 0:
            result.append(symbol)
        previous = symbol
    return tuple(result)


@dataclass(frozen=True)
class EditStep:
    prompt_index: int | None
    decoded_index: int | None
    recognition_edit: str


@dataclass(frozen=True)
class EditAlignment:
    distance: int
    steps: tuple[EditStep, ...]
    ambiguous: bool


def levenshtein(prompt: Sequence[int], decoded: Sequence[int]) -> EditAlignment:
    """Unit edits; diagonal(match/sub), deletion, insertion tie priority.

    Counts optimal paths up to two to expose ambiguity without exponential
    storage. Trace is always deterministic, including repeated prompt phones.
    """
    n, m = len(prompt), len(decoded)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    ways = [[1] * (m + 1) for _ in range(n + 1)]
    ops = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0], ops[i][0] = i, "deletion"
    for j in range(1, m + 1):
        costs[0][j], ops[0][j] = j, "insertion"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = "match" if prompt[i - 1] == decoded[j - 1] else "substitution"
            candidates = [(costs[i - 1][j - 1] + (diagonal != "match"), diagonal,
                           ways[i - 1][j - 1]),
                          (costs[i - 1][j] + 1, "deletion", ways[i - 1][j]),
                          (costs[i][j - 1] + 1, "insertion", ways[i][j - 1])]
            best = min(x[0] for x in candidates)
            costs[i][j] = best
            ops[i][j] = next(x[1] for x in candidates if x[0] == best)
            ways[i][j] = min(2, sum(x[2] for x in candidates if x[0] == best))
    steps, i, j = [], n, m
    while i or j:
        op = ops[i][j]
        steps.append(EditStep(i - 1 if op != "insertion" else None,
                              j - 1 if op != "deletion" else None, op))
        i -= op != "insertion"
        j -= op != "deletion"
    return EditAlignment(costs[n][m], tuple(reversed(steps)), ways[n][m] > 1)


@dataclass(frozen=True)
class ActivationAlignment:
    log_path_mass: float
    states: tuple[int, ...]
    phone_frames: tuple[tuple[int, ...], ...]
    error_scores: tuple[float, ...]
    tied_best_path: bool


def self_aligned_gop(rows: Sequence[Sequence[float]],
                     phones: Sequence[int]) -> ActivationAlignment | None:
    """Canonical CTC Viterbi then negative mean log p(phone) on phone states.

    This explicitly named GOP-SA activation-average control is not SF scoring.
    Blank states excluded, no external intervals, no added probability scale.
    Ties choose lowest predecessor/terminal state; ambiguity is reported.
    """
    states = expanded_states(phones)
    scores = [NEG_INF] * len(states)
    ways = [0] * len(states)
    scores[0], ways[0] = 0.0, 1
    traces = []
    for row in rows:
        current, current_ways, trace = [], [], []
        for s, symbol in enumerate(states):
            incoming = predecessors(states, s)
            best = max(scores[j] for j in incoming)
            tied = [j for j in incoming if scores[j] == best]
            current.append(best + row[symbol])
            current_ways.append(min(2, sum(ways[j] for j in tied)))
            trace.append(min(tied))
        scores, ways = current, current_ways
        traces.append(trace)
    terminals = [len(states) - 1] + ([len(states) - 2] if phones else [])
    best = max(scores[j] for j in terminals)
    if best == NEG_INF:
        return None
    tied = [j for j in terminals if scores[j] == best]
    state = min(tied)
    trace_states = []
    for t in range(len(rows) - 1, -1, -1):
        trace_states.append(state)
        state = traces[t][state]
    trace_states.reverse()
    frames = tuple(tuple(t for t, s in enumerate(trace_states) if s == 2 * i + 1)
                   for i in range(len(phones)))
    errors = tuple(-math.fsum(rows[t][phones[i]] for t in ts) / len(ts)
                   for i, ts in enumerate(frames))
    return ActivationAlignment(best, tuple(trace_states), frames, errors,
                               sum(ways[j] for j in tied) > 1)
