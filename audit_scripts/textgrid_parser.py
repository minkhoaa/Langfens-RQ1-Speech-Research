import re

def parse_textgrid(text):
    """
    Parses a Praat TextGrid string into a dict of tiers.
    Each tier is a dict:
        'class': str,
        'name': str,
        'xmin': float,
        'xmax': float,
        'intervals': list of (xmin, xmax, text)
    """
    lines = [line.strip() for line in text.splitlines()]
    tiers = {}
    i = 0
    n = len(lines)
    
    # We look for item [k]:
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
                    except: pass
                elif l.startswith('xmax =') and 'intervals' not in l:
                    try:
                        tier_xmax = float(l.split('=', 1)[1].strip())
                    except: pass
                elif l.startswith('intervals ['):
                    # parse one interval
                    i += 1
                    int_xmin, int_xmax, int_text = 0.0, 0.0, ""
                    while i < n and not lines[i].startswith('intervals [') and not lines[i].startswith('item ['):
                        il = lines[i]
                        if il.startswith('xmin ='):
                            int_xmin = float(il.split('=', 1)[1].strip())
                        elif il.startswith('xmax ='):
                            int_xmax = float(il.split('=', 1)[1].strip())
                        elif il.startswith('text ='):
                            # text could span multiple lines if quoted, but usually single line
                            int_text = il.split('=', 1)[1].strip()
                            if int_text.startswith('"') and int_text.endswith('"') and len(int_text) >= 2:
                                int_text = int_text[1:-1]
                            elif int_text.startswith('"'):
                                # multi-line text
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

if __name__ == "__main__":
    import zipfile, io
    zf = zipfile.ZipFile('datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip', 'r')
    szf = zipfile.ZipFile(io.BytesIO(zf.read('TLV.zip')))
    tg_text = szf.read('TLV/annotation/arctic_a0003.TextGrid').decode('utf-8')
    tiers = parse_textgrid(tg_text)
    print("Parsed tiers:", list(tiers.keys()))
    print("Phones tier interval count:", len(tiers['phones']['intervals']))
    print("First 10 phones intervals:", tiers['phones']['intervals'][:10])
