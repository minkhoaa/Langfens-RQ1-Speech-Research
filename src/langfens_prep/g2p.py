"""
LANGFENS — Protocol Z: Deterministic CMUdict G2P Frontend
Converts written English prompt sentences into normalized 39-ARPAbet phoneme sequences
without accessing acoustic recordings or human-annotated TextGrids.

Guarantees:
1. Deterministic and reproducible tokenization and dictionary lookup.
2. Cryptographically pinned CMUdict dictionary (v0.7b / master commit 7479086).
3. Explicit out-of-vocabulary (OOV) handling and phonological inflection rules.
4. Output phone vocabulary strictly normalized to standard 39-ARPAbet tokens.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from langfens_prep.phoneme_normalizer import (
    PHONE_TO_CTC_INDEX,
    STANDARD_39_ARPABET,
    normalize_phoneme,
)

EXPECTED_CMUDICT_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
DEFAULT_CMUDICT_PATH = Path("resources/CMUDict/cmudict.dict")

# Standard phonological compounding fallbacks for known composite nouns
KNOWN_COMPOUND_FALLBACKS: Dict[str, List[str]] = {
    "NIGHTGLOW": ["NIGHT", "GLOW"],
    "TOMFOOLERY": ["TOM", "FOOLERY"],
    "ROADMATE": ["ROAD", "MATE"],
    "SEAFARING": ["SEA", "FARING"],
    "UNQUENCHABLE": ["UN", "QUENCH", "ABLE"],
}

# Sibilant consonants triggering /IH Z/ plural/possessive suffix
SIBILANTS: Set[str] = {"S", "Z", "SH", "ZH", "CH", "JH"}
# Voiceless non-sibilant consonants triggering /S/ suffix
VOICELESS_NON_SIBILANTS: Set[str] = {"P", "T", "K", "F", "TH"}


def compute_file_sha256(path: Path | str) -> str:
    """Computes SHA-256 hash of a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class CMUDictG2P:
    """
    Deterministic G2P frontend based on pinned CMUdict.
    Produces zero-oracle canonical phoneme sequences from prompt text.
    """

    def __init__(
        self,
        dict_path: Path | str = DEFAULT_CMUDICT_PATH,
        verify_checksum: bool = True,
        oov_policy: str = "strict",
    ):
        self.dict_path = Path(dict_path)
        self.oov_policy = oov_policy.lower()
        if self.oov_policy not in ("strict", "skip", "error"):
            raise ValueError(f"Unsupported oov_policy: {self.oov_policy}")

        if verify_checksum:
            actual_sha = compute_file_sha256(self.dict_path)
            if actual_sha != EXPECTED_CMUDICT_SHA256:
                raise ValueError(
                    f"CMUdict checksum mismatch!\n"
                    f"  Expected: {EXPECTED_CMUDICT_SHA256}\n"
                    f"  Actual:   {actual_sha}"
                )

        self.entries: Dict[str, List[str]] = {}
        self._load_dictionary()

    def _load_dictionary(self) -> None:
        """Loads primary pronunciations from CMUdict file."""
        with open(self.dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";;;"):
                    continue
                parts = line.split()
                raw_word = parts[0].upper()
                # Exclude alternative pronunciations (word(2), word(3), etc.)
                if "(" in raw_word and raw_word.endswith(")"):
                    continue
                if raw_word not in self.entries:
                    normalized = [
                        normalize_phoneme(p).normalized_phone
                        for p in parts[1:]
                    ]
                    # Validate all phones belong to standard 39-ARPAbet
                    valid_phones = [p for p in normalized if p in STANDARD_39_ARPABET]
                    self.entries[raw_word] = valid_phones

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes written prompt text into clean uppercase word tokens.
        Preserves contractions and internal apostrophes while stripping punctuation.
        """
        # Normalize double hyphens and dashes to spaces
        cleaned = text.replace("--", " ").replace("—", " ").replace("–", " ")
        # Regex extracts alphanumeric words with optional internal apostrophe or leading apostrophe for contractions
        raw_tokens = re.findall(r"(?:'?[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?)", cleaned)
        tokens = []
        for t in raw_tokens:
            w = t.upper().strip(".,;:!?\"'()")
            # If word was a leading-apostrophe contraction (e.g. 'em, 'cause, 'round)
            if t.startswith("'") and t.upper() in self.entries:
                tokens.append(t.upper())
            elif w:
                tokens.append(w)
        return tokens

    def _inflect_possessive(self, base_phones: List[str]) -> List[str]:
        """Applies English regular possessive/plural phonological inflection rules."""
        if not base_phones:
            return ["Z"]
        last_phone = base_phones[-1]
        if last_phone in SIBILANTS:
            return base_phones + ["IH", "Z"]
        elif last_phone in VOICELESS_NON_SIBILANTS:
            return base_phones + ["S"]
        else:
            return base_phones + ["Z"]

    def word_to_phones(self, word: str) -> List[str]:
        """
        Looks up or derives canonical 39-ARPAbet phones for a single uppercase word.
        """
        # 1. Direct dictionary lookup
        if word in self.entries:
            return list(self.entries[word])

        # 2. Outer apostrophe strip
        w_strip = word.strip("'")
        if w_strip in self.entries:
            return list(self.entries[w_strip])

        # 3. Known compound words
        if word in KNOWN_COMPOUND_FALLBACKS:
            phones = []
            for subw in KNOWN_COMPOUND_FALLBACKS[word]:
                phones.extend(self.entries[subw])
            return phones
        if w_strip in KNOWN_COMPOUND_FALLBACKS:
            phones = []
            for subw in KNOWN_COMPOUND_FALLBACKS[w_strip]:
                phones.extend(self.entries[subw])
            return phones

        # 4. Possessive 'S handling
        if word.endswith("'S"):
            base = word[:-2]
            if base in self.entries:
                return self._inflect_possessive(self.entries[base])
            # Check if base without apostrophe exists (e.g. word without 's)
            base_plural = base + "S"
            if base_plural in self.entries:
                return list(self.entries[base_plural])

        # 5. Hyphenated compound split
        if "-" in word:
            subparts = word.split("-")
            phones = []
            all_found = True
            for sp in subparts:
                sp_clean = sp.strip("'")
                if sp_clean in self.entries:
                    phones.extend(self.entries[sp_clean])
                else:
                    all_found = False
                    break
            if all_found and phones:
                return phones

        # 6. OOV policy execution
        if self.oov_policy in ("strict", "error"):
            raise KeyError(f"Out-of-vocabulary word '{word}' in CMUdict G2P frontend.")
        elif self.oov_policy == "skip":
            return []
        else:
            raise KeyError(f"Unhandled OOV word: {word}")

    def phonemize(self, text: str) -> List[str]:
        """
        Transforms written prompt text into canonical 39-ARPAbet phone sequence.
        """
        tokens = self.tokenize(text)
        phones = []
        for tok in tokens:
            word_phones = self.word_to_phones(tok)
            phones.extend(word_phones)
        return phones

    def encode_indices(self, text: str) -> List[int]:
        """
        Transforms written prompt text into 1-indexed CTC token IDs.
        """
        phones = self.phonemize(text)
        return [PHONE_TO_CTC_INDEX[p] for p in phones]
