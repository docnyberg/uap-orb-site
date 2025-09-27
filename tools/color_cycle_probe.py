#!/usr/bin/env python3
import json, sys
from pathlib import Path
from collections import Counter

def strip_all(tok):
    t=tok[:-1] if tok.endswith("✦") else tok
    if t.endswith("↑") or t.endswith("↓"): t=t[:-1]
    return t

def main(seq_dir):
    counts = Counter()
    for p in Path(seq_dir).glob("*.sequence.json"):
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        cycles = data.get("cycles") or [data.get("symbol_seq_enhanced","")]
        for cyc in cycles:
            toks=[strip_all(t) for t in (cyc or "").split() if t]
            # run-length collapse
            r=[]
            for t in toks:
                if not r or t!=r[-1]: r.append(t)
            # sliding windows length 4 with 4 distinct colors
            for i in range(len(r)-3):
                win=r[i:i+4]
                if len(set(win))==4:
                    counts[tuple(win)]+=1
    for seq, n in counts.most_common(20):
        print(n, " ", " → ".join(seq))

if __name__=="__main__":
    main(sys.argv[1])
