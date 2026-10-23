import pytest
from langfens_prep.phoneme_normalizer import (
    normalize_phoneme,
    arpabet_to_ipa,
    STANDARD_39_ARPABET,
    SILENCE_SYMBOLS
)

def test_standard_phoneme_normalization():
    res = normalize_phoneme("AA1")
    assert res.normalized_phone == "AA"
    assert res.stress == 1
    assert res.raw_token == "AA1"
    assert res.has_asterisk is False
    assert res.is_err is False

def test_ax_to_ah_folding():
    res = normalize_phoneme("AX0", fold_ax_to_ah=True)
    assert res.normalized_phone == "AH"
    assert res.was_ax is True
    assert res.stress == 0

    res_no_fold = normalize_phoneme("AX0", fold_ax_to_ah=False)
    assert res_no_fold.normalized_phone == "AX"
    assert res_no_fold.was_ax is True

def test_asterisk_and_err_flags():
    res_star = normalize_phoneme("T*")
    assert res_star.normalized_phone == "T"
    assert res_star.has_asterisk is True

    res_err = normalize_phoneme("err")
    assert res_err.is_err is True
    assert res_err.normalized_phone == "<ERR>"

def test_typo_correction_v8():
    res = normalize_phoneme("V8", correct_typos=True)
    assert res.normalized_phone == "V"
    assert res.is_typo_correction is True
    assert res.raw_token == "V8"

def test_reversibility():
    original = "ER0*"
    res = normalize_phoneme(original)
    reconstructed = res.reconstruct()
    assert reconstructed == original

def test_ipa_mapping():
    ipa = arpabet_to_ipa("SH")
    assert ipa == "ʃ"
    ipa_th = arpabet_to_ipa("TH")
    assert ipa_th == "θ"
