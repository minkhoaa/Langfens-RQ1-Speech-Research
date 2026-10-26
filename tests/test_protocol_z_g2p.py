import pytest
from pathlib import Path

from langfens_prep.g2p import (
    CMUDictG2P,
    DEFAULT_CMUDICT_PATH,
    EXPECTED_CMUDICT_SHA256,
    compute_file_sha256,
)
from langfens_prep.phoneme_normalizer import STANDARD_39_ARPABET, PHONE_TO_CTC_INDEX


def test_cmudict_provenance_and_checksum():
    """Verifies that the pinned CMUdict dictionary exists and matches its frozen SHA-256 digest."""
    assert DEFAULT_CMUDICT_PATH.is_file(), f"CMUdict not found at {DEFAULT_CMUDICT_PATH}"
    actual_sha = compute_file_sha256(DEFAULT_CMUDICT_PATH)
    assert actual_sha == EXPECTED_CMUDICT_SHA256, (
        f"CMUdict digest mismatch! Expected {EXPECTED_CMUDICT_SHA256}, got {actual_sha}"
    )


def test_g2p_deterministic_output():
    """Verifies that G2P frontend produces bit-for-bit identical sequences across multiple runs."""
    g2p = CMUDictG2P(verify_checksum=True, oov_policy="strict")
    sample_text = "For the twentieth time that evening the two men shook hands."

    out1 = g2p.phonemize(sample_text)
    out2 = g2p.phonemize(sample_text)
    assert out1 == out2
    assert len(out1) == 39

    idx1 = g2p.encode_indices(sample_text)
    idx2 = g2p.encode_indices(sample_text)
    assert idx1 == idx2
    assert len(idx1) == 39
    assert all(1 <= i <= 39 for i in idx1)


def test_stable_vocabulary_and_phoneme_normalization():
    """Verifies that all phonemes produced by G2P belong to the standard 39-ARPAbet inventory."""
    g2p = CMUDictG2P(verify_checksum=False, oov_policy="strict")
    for word, phones in g2p.entries.items():
        for p in phones:
            assert p in STANDARD_39_ARPABET, f"Non-standard phone '{p}' in word '{word}'"
            assert p in PHONE_TO_CTC_INDEX, f"Phone '{p}' missing from CTC index mapping"


def test_contractions_and_apostrophe_handling():
    """Verifies that common contractions and colloquial forms are correctly resolved."""
    g2p = CMUDictG2P(verify_checksum=False, oov_policy="strict")

    # Contractions in dictionary
    assert g2p.phonemize("don't") == ["D", "OW", "N", "T"]
    assert g2p.phonemize("I'll") == ["AY", "L"]
    assert g2p.phonemize("that's") == ["DH", "AE", "T", "S"]
    assert g2p.phonemize("'em") == ["AH", "M"]
    assert g2p.phonemize("'round") == ["R", "AW", "N", "D"]


def test_possessive_phonological_inflection():
    """Verifies phonological voicing assimilation for English possessive 's."""
    g2p = CMUDictG2P(verify_checksum=False, oov_policy="strict")

    # 1. Base ending in voiced non-sibilant -> Z (e.g. promoter ending in ER)
    assert g2p.word_to_phones("PROMOTER'S") == ["P", "R", "AH", "M", "OW", "T", "ER", "Z"]

    # 2. Base ending in voiceless non-sibilant -> S (e.g. cat -> cats/cat's)
    assert g2p.word_to_phones("CAT'S") == ["K", "AE", "T", "S"]

    # 3. Direct inflection unit test for sibilant coda -> IH Z
    assert g2p._inflect_possessive(["B", "AA", "SH"]) == ["B", "AA", "SH", "IH", "Z"]
    assert g2p._inflect_possessive(["F", "IH", "Z"]) == ["F", "IH", "Z", "IH", "Z"]


def test_compound_and_hyphen_decomposition():
    """Verifies compound splitting for hyphenated and known unhyphenated ARCTIC words."""
    g2p = CMUDictG2P(verify_checksum=False, oov_policy="strict")

    # Hyphenated
    assert g2p.phonemize("wolf-dog") == ["W", "UH", "L", "F", "D", "AO", "G"]
    assert g2p.phonemize("smooth-shaven") == ["S", "M", "UW", "DH", "SH", "EY", "V", "AH", "N"]

    # Known unhyphenated
    assert g2p.phonemize("nightglow") == ["N", "AY", "T", "G", "L", "OW"]
    assert g2p.phonemize("tomfoolery") == ["T", "AA", "M", "F", "UW", "L", "ER", "IY"]


def test_explicit_oov_behavior():
    """Verifies that OOV policy behaves explicitly and strictly."""
    g2p_strict = CMUDictG2P(verify_checksum=False, oov_policy="strict")
    with pytest.raises(KeyError) as exc_info:
        g2p_strict.phonemize("Supercalifragilisticexpialidocious")
    assert "Out-of-vocabulary word" in str(exc_info.value)

    g2p_skip = CMUDictG2P(verify_checksum=False, oov_policy="skip")
    assert g2p_skip.phonemize("Supercalifragilisticexpialidocious") == []
