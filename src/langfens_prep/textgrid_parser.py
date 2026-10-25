import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any

from langfens_prep.phoneme_normalizer import (
    normalize_phoneme,
    SILENCE_SYMBOLS,
    STANDARD_39_ARPABET
)

@dataclass
class TextGridInterval:
    token_index: Optional[int]
    xmin: float
    xmax: float
    raw_label: str
    error_type: str  # "correct", "substitution", "deletion", "addition", "silence"
    is_canonical: bool
    canonical_phone: Optional[str]
    realized_phone: Optional[str]
    canonical_stress: Optional[int]
    realized_stress: Optional[int]
    is_base_preserving: bool
    has_asterisk: bool
    has_err_tag: bool
    is_typo_correction: bool
    binary_label_strict: Optional[int]  # 0 or 1 for canonical tokens, None for non-canonical
    binary_label_lenient: Optional[int]
    diagnosis_target: Optional[str]  # 39-ARPAbet substitute phone for eligible substitutions; None for correct/deletion/err/masked

@dataclass
class L2ArcticUtteranceAnnotation:
    filename: str
    intervals: List[TextGridInterval]
    words_intervals: List[Dict[str, Any]]
    total_duration: float

    @property
    def canonical_intervals(self) -> List[TextGridInterval]:
        return [i for i in self.intervals if i.is_canonical]

def parse_textgrid_raw_tiers(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses a Praat TextGrid string into a dict of tiers.
    Extracts class, name, xmin, xmax, and interval tuples (xmin, xmax, text).
    """
    lines = [line.strip() for line in text.splitlines()]
    tiers = {}
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i]
        if line.startswith('item ['):
            i += 1
            tier_class = ""
            tier_name = ""
            tier_xmin = 0.0
            tier_xmax = 0.0
            intervals = []
            
            while i < n and not lines[i].startswith('item ['):
                l = lines[i]
                if l.startswith('class ='):
                    tier_class = l.split('=', 1)[1].strip().strip('"')
                elif l.startswith('name ='):
                    tier_name = l.split('=', 1)[1].strip().strip('"')
                elif l.startswith('xmin =') and 'intervals' not in l:
                    try:
                        tier_xmin = float(l.split('=', 1)[1].strip())
                    except ValueError:
                        pass
                elif l.startswith('xmax =') and 'intervals' not in l:
                    try:
                        tier_xmax = float(l.split('=', 1)[1].strip())
                    except ValueError:
                        pass
                elif l.startswith('intervals ['):
                    i += 1
                    int_xmin, int_xmax, int_text = 0.0, 0.0, ""
                    while i < n and not lines[i].startswith('intervals [') and not lines[i].startswith('item ['):
                        il = lines[i]
                        if il.startswith('xmin ='):
                            int_xmin = float(il.split('=', 1)[1].strip())
                        elif il.startswith('xmax ='):
                            int_xmax = float(il.split('=', 1)[1].strip())
                        elif il.startswith('text ='):
                            int_text = il.split('=', 1)[1].strip()
                            if int_text.startswith('"') and int_text.endswith('"') and len(int_text) >= 2:
                                int_text = int_text[1:-1]
                            elif int_text.startswith('"'):
                                int_text = int_text[1:]
                                while i + 1 < n and not lines[i+1].endswith('"'):
                                    i += 1
                                    int_text += "\n" + lines[i]
                                if i + 1 < n:
                                    i += 1
                                    int_text += "\n" + lines[i][:-1]
                        i += 1
                    intervals.append((int_xmin, int_xmax, int_text))
                    continue
                i += 1
            if tier_name:
                tiers[tier_name] = {
                    'class': tier_class,
                    'name': tier_name,
                    'xmin': tier_xmin,
                    'xmax': tier_xmax,
                    'intervals': intervals
                }
            continue
        i += 1
    return tiers

def parse_gold_textgrid_content(text: str, filename: str) -> L2ArcticUtteranceAnnotation:
    """
    Parses gold manual Praat TextGrid content adhering strictly to RQ1-STD-01
    and CORRECTED_LABEL_ACCOUNTING specifications.
    """
    tiers = parse_textgrid_raw_tiers(text)
    if 'phones' not in tiers:
        raise ValueError(f"TextGrid {filename} missing required 'phones' tier")
        
    phones_tier = tiers['phones']
    words_tier = tiers.get('words', {'intervals': []})
    
    total_duration = phones_tier['xmax']
    parsed_intervals: List[TextGridInterval] = []
    
    canonical_token_idx = 0
    
    for xmin, xmax, raw_label in phones_tier['intervals']:
        trimmed = raw_label.strip()
        
        # 1. Silence check
        if trimmed.lower() in SILENCE_SYMBOLS:
            parsed_intervals.append(TextGridInterval(
                token_index=None,
                xmin=xmin,
                xmax=xmax,
                raw_label=raw_label,
                error_type="silence",
                is_canonical=False,
                canonical_phone=None,
                realized_phone="sil",
                canonical_stress=None,
                realized_stress=None,
                is_base_preserving=False,
                has_asterisk=False,
                has_err_tag=False,
                is_typo_correction=False,
                binary_label_strict=None,
                binary_label_lenient=None,
                diagnosis_target=None
            ))
            continue
            
        # 2. Comma tag (substitutions, deletions, additions)
        if ',' in trimmed:
            parts = [p.strip() for p in trimmed.split(',')]
            if len(parts) == 3:
                cpl, ppl, err_code = parts[0], parts[1], parts[2].lower()
                has_star = ('*' in cpl or '*' in ppl)
                has_err = ('err' in cpl.lower() or 'err' in ppl.lower())
                
                cpl_norm = normalize_phoneme(cpl)
                ppl_norm = normalize_phoneme(ppl)
                
                is_typo = cpl_norm.is_typo_correction or ppl_norm.is_typo_correction
                
                if err_code == 's':
                    # Substitution interval
                    base_c = cpl_norm.normalized_phone
                    base_p = ppl_norm.normalized_phone
                    is_same_base = (base_c == base_p)
                    
                    # Strict policy: ALL substitutions are Errors (1)
                    bin_strict = 1
                    
                    # Lenient policy: Base-preserving allophones (CPL, CPL*, s) become Correct (0)
                    if is_same_base and has_star:
                        bin_lenient = 0
                    else:
                        bin_lenient = 1
                        
                    # Diagnosis target: masked for 'err' tags; valid 39-phone otherwise
                    if has_err or base_p not in STANDARD_39_ARPABET:
                        diag_target = None
                    else:
                        diag_target = base_p
                        
                    parsed_intervals.append(TextGridInterval(
                        token_index=canonical_token_idx,
                        xmin=xmin,
                        xmax=xmax,
                        raw_label=raw_label,
                        error_type="substitution",
                        is_canonical=True,
                        canonical_phone=base_c,
                        realized_phone=base_p,
                        canonical_stress=cpl_norm.stress,
                        realized_stress=ppl_norm.stress,
                        is_base_preserving=is_same_base,
                        has_asterisk=has_star,
                        has_err_tag=has_err,
                        is_typo_correction=is_typo,
                        binary_label_strict=bin_strict,
                        binary_label_lenient=bin_lenient,
                        diagnosis_target=diag_target
                    ))
                    canonical_token_idx += 1
                    
                elif err_code == 'd':
                    # Deletion interval
                    base_c = cpl_norm.normalized_phone
                    parsed_intervals.append(TextGridInterval(
                        token_index=canonical_token_idx,
                        xmin=xmin,
                        xmax=xmax,
                        raw_label=raw_label,
                        error_type="deletion",
                        is_canonical=True,
                        canonical_phone=base_c,
                        realized_phone="sil",
                        canonical_stress=cpl_norm.stress,
                        realized_stress=None,
                        is_base_preserving=False,
                        has_asterisk=has_star,
                        has_err_tag=has_err,
                        is_typo_correction=is_typo,
                        binary_label_strict=1,
                        binary_label_lenient=1,
                        diagnosis_target=None  # Deletions have no realized phone target
                    ))
                    canonical_token_idx += 1
                    
                elif err_code == 'a':
                    # Epenthetic addition / insertion (unprompted, excluded from canonical prompt denominator)
                    base_p = ppl_norm.normalized_phone
                    parsed_intervals.append(TextGridInterval(
                        token_index=None,
                        xmin=xmin,
                        xmax=xmax,
                        raw_label=raw_label,
                        error_type="addition",
                        is_canonical=False,
                        canonical_phone=None,
                        realized_phone=base_p,
                        canonical_stress=None,
                        realized_stress=ppl_norm.stress,
                        is_base_preserving=False,
                        has_asterisk=has_star,
                        has_err_tag=has_err,
                        is_typo_correction=is_typo,
                        binary_label_strict=None,
                        binary_label_lenient=None,
                        diagnosis_target=None
                    ))
                else:
                    raise ValueError(f"Unknown error tag '{err_code}' in interval: {raw_label}")
            else:
                raise ValueError(f"Malformed comma tag (expected 3 parts): {raw_label}")
        else:
            # 3. Single phone segment -> Correct production
            norm = normalize_phoneme(trimmed)
            base_phone = norm.normalized_phone
            parsed_intervals.append(TextGridInterval(
                token_index=canonical_token_idx,
                xmin=xmin,
                xmax=xmax,
                raw_label=raw_label,
                error_type="correct",
                is_canonical=True,
                canonical_phone=base_phone,
                realized_phone=base_phone,
                canonical_stress=norm.stress,
                realized_stress=norm.stress,
                is_base_preserving=True,
                has_asterisk=False,
                has_err_tag=False,
                is_typo_correction=norm.is_typo_correction,
                binary_label_strict=0,
                binary_label_lenient=0,
                diagnosis_target=None  # Correct productions have no substitution diagnosis target
            ))
            canonical_token_idx += 1
            
    words_parsed = [
        {"xmin": w[0], "xmax": w[1], "text": w[2]}
        for w in words_tier['intervals']
    ]
    
    return L2ArcticUtteranceAnnotation(
        filename=filename,
        intervals=parsed_intervals,
        words_intervals=words_parsed,
        total_duration=total_duration
    )
