#!/usr/bin/env python3
import json, sys
from pathlib import Path
from collections import defaultdict, Counter

def toks(s): return [t for t in (s or "").split() if t]

def main(seq_dir):
    per_color = defaultdict(lambda: Counter(toggles=0, stays=0, total=0))
    for p in Path(seq_dir).glob("*.sequence.json"):
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        ts = toks(data.get("symbol_seq_enhanced",""))
        # compare consecutive tokens when same color base
        def base(t):  # strip ↑/↓/✦ and return the color word
            x=t[:-1] if t.endswith("✦") else t
            if x.endswith("↑") or x.endswith("↓"): x=x[:-1]
            return x
        def ori(t):  # returns '↑'/'↓'/None
            tt=t[:-1] if t.endswith("✦") else t
            return tt[-1] if tt.endswith("↑") or tt.endswith("↓") else None

        for a,b in zip(ts, ts[1:]):
            if base(a)==base(b):
                c=base(a)
                oa,ob=ori(a),ori(b)
                if oa and ob and oa!=ob:
                    per_color[c]['toggles']+=1
                else:
                    per_color[c]['stays']+=1
                per_color[c]['total']+=1
    for c, cnt in per_color.items():
        t=cnt['toggles']; s=cnt['stays']; tot=cnt['total'] or 1
        print(f"{c:15s}  toggle%={t*100/tot:5.1f}  stay%={s*100/tot:5.1f}  n={tot}")

if __name__=="__main__":
    main(sys.argv[1])
