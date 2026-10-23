import re
from dataclasses import dataclass
from typing import Optional, Dict

STANDARD_39_ARPABET = (
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", 
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", 
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", 
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
)

# Standard index mapping for CTC targets: index 0 is reserved strictly for <BLANK>
PHONE_TO_CTC_INDEX: Dict[str, int] = {phone: idx + 1 for idx, phone in enumerate(STANDARD_39_ARPABET)}
CTC_INDEX_TO_PHONE: Dict[int, str] = {idx + 1: phone for idx, phone in enumerate(STANDARD_39_ARPABET)}
CTC_INDEX_TO_PHONE[0] = "<BLANK>"

SILENCE_SYMBOLS = {"", "sil", "sp", "spn", "silence", "<sil>", "<pau>"}

ARPABET_TO_IPA: Dict[str, str] = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ"
}

@dataclass
class PhonemeNormalizationResult:
    raw_token: str
    normalized_phone: str
    stress: Optional[int]
    has_asterisk: bool
    is_err: bool
    was_ax: bool
    is_typo_correction: bool

    def reconstruct(self) -> str:
        """Reconstructs the original raw token from saved transformation metadata."""
        if self.is_err:
            return self.raw_token
        if self.is_typo_correction:
            return self.raw_token
        base = "AX" if self.was_ax else self.normalized_phone
        stress_str = str(self.stress) if self.stress is not None else ""
        star_str = "*" if self.has_asterisk else ""
        return f"{base}{stress_str}{star_str}"

def normalize_phoneme(
    token: str, 
    fold_ax_to_ah: bool = True, 
    strip_stress: bool = True,
    correct_typos: bool = True
) -> PhonemeNormalizationResult:
    """
    Non-destructively normalizes an ARPAbet phonetic token while recording all
    transformation metadata to allow exact reversibility.
    """
    raw = token
    trimmed = token.strip()
    
    if trimmed.lower() == "err":
        return PhonemeNormalizationResult(
            raw_token=raw,
            normalized_phone="<ERR>",
            stress=None,
            has_asterisk=False,
            is_err=True,
            was_ax=False,
            is_typo_correction=False
        )
    
    # Check typo correction: e.g. SVBI keyboard slip 'V8' for 'V*' or 'V'
    is_typo = False
    if correct_typos and trimmed == "V8":
        is_typo = True
        trimmed = "V"
        
    has_asterisk = "*" in trimmed
    cleaned = trimmed.replace("*", "")
    
    # Extract stress digit (0, 1, 2)
    stress = None
    stress_match = re.search(r"(\d+)$", cleaned)
    if stress_match:
        stress = int(stress_match.group(1))
        base_phone = cleaned[:stress_match.start()].upper()
    else:
        base_phone = cleaned.upper()
        
    was_ax = (base_phone == "AX")
    if was_ax and fold_ax_to_ah:
        norm_phone = "AH"
    else:
        norm_phone = base_phone
        
    return PhonemeNormalizationResult(
        raw_token=raw,
        normalized_phone=norm_phone,
        stress=stress,
        has_asterisk=has_asterisk,
        is_err=False,
        was_ax=was_ax,
        is_typo_correction=is_typo
    )

def arpabet_to_ipa(phone: str) -> str:
    """Maps a normalized 39-ARPAbet phone to standard IPA representation."""
    norm = normalize_phoneme(phone).normalized_phone
    return ARPABET_TO_IPA.get(norm, phone)
