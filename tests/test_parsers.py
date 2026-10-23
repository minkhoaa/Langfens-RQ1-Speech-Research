import pytest
from langfens_prep.textgrid_parser import parse_gold_textgrid_content, TextGridInterval
from langfens_prep.speechocean_parser import parse_speechocean_manifested

SAMPLE_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 1.5 
tiers? <exists> 
size = 2 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "words" 
        xmin = 0 
        xmax = 1.5 
        intervals: size = 2 
        intervals [1]:
            xmin = 0 
            xmax = 0.5 
            text = "sil" 
        intervals [2]:
            xmin = 0.5 
            xmax = 1.5 
            text = "the" 
    item [2]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 1.5 
        intervals: size = 6 
        intervals [1]:
            xmin = 0 
            xmax = 0.5 
            text = "sil" 
        intervals [2]:
            xmin = 0.5 
            xmax = 0.7 
            text = "DH,D,s" 
        intervals [3]:
            xmin = 0.7 
            xmax = 0.9 
            text = "AH0" 
        intervals [4]:
            xmin = 0.9 
            xmax = 1.1 
            text = "R,R*,s" 
        intervals [5]:
            xmin = 1.1 
            xmax = 1.3 
            text = "T,sil,d" 
        intervals [6]:
            xmin = 1.3 
            xmax = 1.5 
            text = "sil,AH,a" 
"""

def test_parse_gold_textgrid_intervals():
    utt = parse_gold_textgrid_content(SAMPLE_TEXTGRID, "arctic_test.TextGrid")
    assert len(utt.intervals) == 6
    assert len(utt.canonical_intervals) == 4 # DH, AH0, R, T (silence and addition excluded)
    
    # Interval 1: silence
    assert utt.intervals[0].error_type == "silence"
    assert utt.intervals[0].is_canonical is False
    
    # Interval 2: substitution DH -> D
    dh_interval = utt.intervals[1]
    assert dh_interval.error_type == "substitution"
    assert dh_interval.canonical_phone == "DH"
    assert dh_interval.realized_phone == "D"
    assert dh_interval.binary_label_strict == 1
    assert dh_interval.binary_label_lenient == 1
    
    # Interval 3: correct AH0
    ah_interval = utt.intervals[2]
    assert ah_interval.error_type == "correct"
    assert ah_interval.canonical_phone == "AH"
    assert ah_interval.realized_phone == "AH"
    assert ah_interval.binary_label_strict == 0
    assert ah_interval.binary_label_lenient == 0
    
    # Interval 4: base-preserving allophone R -> R*
    r_interval = utt.intervals[3]
    assert r_interval.error_type == "substitution"
    assert r_interval.is_base_preserving is True
    assert r_interval.binary_label_strict == 1
    assert r_interval.binary_label_lenient == 0 # Lenient reclassifies as correct!
    
    # Interval 5: deletion T -> sil
    t_interval = utt.intervals[4]
    assert t_interval.error_type == "deletion"
    assert t_interval.binary_label_strict == 1
    assert t_interval.binary_label_lenient == 1
    
    # Interval 6: addition sil -> AH
    add_interval = utt.intervals[5]
    assert add_interval.error_type == "addition"
    assert add_interval.is_canonical is False

def test_speechocean_parser_manifested():
    archive_path = "datasets/SpeechOcean762/original/speechocean762.tar.gz"
    split_manifest = "manifests/RQ1-SPLIT-01/speechocean762_official_splits.json"
    
    records = parse_speechocean_manifested(archive_path, split_manifest, limit=10)
    assert len(records) == 10
    first = next(iter(records.values()))
    assert first.utterance_id is not None
    assert first.speaker_id is not None
    assert 0.0 <= first.overall_accuracy <= 10.0
    assert len(first.words) > 0
    assert len(first.words[0].phones) > 0
    assert len(first.words[0].phones_accuracy) == len(first.words[0].phones)
