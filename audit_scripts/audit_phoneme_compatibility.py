import zipfile
import tarfile
import json
import re
from collections import Counter

# 1. L2-ARCTIC symbols
with open("audit_scripts/l2arctic_audit_summary.json") as f:
    l2_summary = json.load(f)

# Collect all unique raw and clean symbols from L2-ARCTIC
zf = zipfile.ZipFile("datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip", "r")
from textgrid_parser import parse_textgrid
import io

l2_canonical_symbols = set()
l2_realized_symbols = set()
l2_special_tokens = set()

# sample speakers across all L1s
speakers = ['ABA', 'BWC', 'ASI', 'HJK', 'EBVS', 'TLV']
for spk in speakers:
    szf = zipfile.ZipFile(io.BytesIO(zf.read(f"{spk}.zip")))
    for n in szf.namelist():
        if n.startswith(f"{spk}/annotation/") and n.endswith(".TextGrid"):
            tiers = parse_textgrid(szf.read(n).decode('utf-8'))
            if 'phones' in tiers:
                for xmin, xmax, label in tiers['phones']['intervals']:
                    lbl = label.strip()
                    if not lbl: continue
                    if ',' in lbl:
                        parts = [p.strip() for p in lbl.split(',')]
                        if len(parts) == 3:
                            cpl, ppl, et = parts
                            l2_canonical_symbols.add(cpl)
                            l2_realized_symbols.add(ppl)
                            if '*' in lbl or 'err' in lbl:
                                l2_special_tokens.add(lbl)
                    else:
                        l2_canonical_symbols.add(lbl)
zf.close()

# 2. CMUDict symbols
with open("resources/CMUDict/cmudict.symbols") as f:
    cmu_symbols = set(line.strip() for line in f if line.strip())

with open("resources/CMUDict/cmudict.phones") as f:
    cmu_phones = set(line.strip().split()[0] for line in f if line.strip())

# 3. MFA english_us_arpa dictionary symbols
mfa_phones = set()
with open("resources/MFA/dictionaries/english_us_arpa.dict") as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if len(parts) > 1:
            for p in parts[1:]:
                mfa_phones.add(p)
        if i > 50000: # sample first 50k lines
            pass

# 4. SpeechOcean762 symbols
tf = tarfile.open("datasets/SpeechOcean762/original/speechocean762.tar.gz", "r:gz")
so_scores = json.loads(tf.extractfile(tf.getmember('speechocean762/resource/scores.json')).read().decode('utf-8'))
so_phones = set()
for utt in so_scores.values():
    for w in utt.get('words', []):
        so_phones.update(w.get('phones', []))
tf.close()

# 5. Print Inventory Comparison
print("=== INVENTORY COMPARISON ===")
print(f"CMUDict base phones (cmudict.phones): {len(cmu_phones)}")
print(f"CMUDict symbols with stress (cmudict.symbols): {len(cmu_symbols)}")
print(f"MFA english_us_arpa phone types: {len(mfa_phones)}")
print(f"SpeechOcean762 phone types in scores.json: {len(so_phones)}")

# Base phone comparison (stripped stress)
def strip_all(s_set):
    res = set()
    for s in s_set:
        c = re.sub(r'[0-9*`_()\[\]{}]', '', s).strip().upper()
        if c and c not in {'SIL', 'SP', 'SPN', 'ERR'}:
            res.add(c)
    return res

cmu_base = strip_all(cmu_phones)
mfa_base = strip_all(mfa_phones)
so_base = strip_all(so_phones)
l2_can_base = strip_all(l2_canonical_symbols)
l2_real_base = strip_all(l2_realized_symbols)

print("\nBase Phone Counts (excluding stress and silences):")
print(f"  CMUDict: {len(cmu_base)} -> {sorted(cmu_base)}")
print(f"  MFA ARPA: {len(mfa_base)} -> {sorted(mfa_base)}")
print(f"  SpeechOcean762: {len(so_base)} -> {sorted(so_base)}")
print(f"  L2-ARCTIC canonical: {len(l2_can_base)} -> {sorted(l2_can_base)}")
print(f"  L2-ARCTIC realized: {len(l2_real_base)} -> {sorted(l2_real_base)}")

print("\nDifferences:")
print("  In L2-ARCTIC canonical but not CMUDict base:", l2_can_base - cmu_base)
print("  In CMUDict base but not L2-ARCTIC canonical:", cmu_base - l2_can_base)
print("  In SpeechOcean762 but not CMUDict base:", so_base - cmu_base)
print("  In CMUDict base but not SpeechOcean762:", cmu_base - so_base)
print("  In MFA ARPA but not CMUDict base:", mfa_base - cmu_base)
print("  In CMUDict base but not MFA ARPA:", cmu_base - mfa_base)

# Check stress representation in SpeechOcean762 vs L2-ARCTIC
print("\nStress Representation:")
print("  Sample SpeechOcean762 phones:", sorted(list(so_phones))[:15])
sample_l2_with_stress = [s for s in l2_canonical_symbols if any(char.isdigit() for char in s)][:15]
print("  Sample L2-ARCTIC canonical phones with stress:", sample_l2_with_stress)
