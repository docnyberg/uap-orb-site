#!/usr/bin/env python3
import json, sys
from pathlib import Path
from collections import defaultdict, Counter

def main(seq_dir):
    pos = Counter()
    total = 0
    for p in Path(seq_dir).glob("*.sequence.json"):
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        cycles = data.get("cycles") or [data.get("symbol_seq_enhanced","")]
        for cyc in cycles:
            toks = [t for t in (cyc or "").split() if t]
            if len(toks)<2: continue
            total += 1
            # mark where flash occurs relative to cycle length
            L=len(toks)
            for i,t in enumerate(toks):
                if t.endswith("✦"):
                    pos[(i,L)] += 1
    # summarize by normalized position (i/L rounded to decile)
    bucket=defaultdict(int)
    for (i,L),c in pos.items():
        r = round(i/max(1,L-1),1)
        bucket[r]+=c
    for r,c in sorted(bucket.items()):
        print(f"pos≈{r:0.1f}  count={c}")
if __name__=="__main__":
    main(sys.argv[1])
