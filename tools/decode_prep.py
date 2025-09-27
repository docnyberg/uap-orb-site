#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_prep.py
Build per-cycle decoding tables from *.sequence.json produced by build_sequences.py.

- RLE color stream per cycle (strip ↑/↓ and ✦)
- Learn the dominant 4-color cyclic order across the dataset (rotation-invariant)
- Rotate each cycle's RLE colors to that canonical order (when possible)
- Record orientation at cycle start (↑/↓/None)
- Record flash positions (absolute index, normalized index) and color-phase hit
- Optionally emit base-4 encoding when the rotated cycle maps cleanly to the canonical order

Usage:
  python decode_prep.py ^
    --sequences "C:\...\public\sequences" ^
    --out-csv  "C:\...\public\decode_prep_cycles.csv" ^
    --out-json "C:\...\public\decode_prep_summary.json" ^
    [--prefer-order "Green,Yellow,Pink/Magenta,Blue"]   # optional fixed order if you want to force it

Notes:
- Works only on the 'cycles' you already emit in each *.sequence.json.
- Strips orientation and flash for order learning; stores them separately for analysis.
"""

import argparse, csv, json, math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Any

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def toks(s: str) -> List[str]:
    return [t for t in (s or "").split() if t]

def strip_all(tok: str) -> str:
    """Strip trailing ✦ and ↑/↓ to get the base color token."""
    t = tok[:-1] if tok.endswith("✦") else tok
    if t.endswith("↑") or t.endswith("↓"):
        t = t[:-1]
    return t

def rle_collapse(seq: List[str]) -> List[str]:
    out = []
    for t in seq:
        if not out or t != out[-1]:
            out.append(t)
    return out

def all_rotations(lst: List[str]) -> List[List[str]]:
    return [lst[i:] + lst[:i] for i in range(len(lst))] if lst else [lst]

def rotation_index_to_match(target: List[str], pattern: List[str]) -> Optional[int]:
    """Return rotation index r s.t. rotate(pattern,r) == target, else None."""
    if len(target) != len(pattern):
        return None
    for r in range(len(pattern)):
        if all(pattern[(i + r) % len(pattern)] == target[i] for i in range(len(pattern))):
            return r
    return None

def best_four_color_order(cycles_rle: List[List[str]], prefer: Optional[List[str]]) -> Optional[List[str]]:
    """
    From many RLE cycles, collect all length-4 windows of 4 distinct colors (rotation-invariant)
    and pick the most frequent cyclic order. If 'prefer' given, return it when observed, otherwise
    fall back to the most frequent in the data.
    """
    counts = Counter()
    seen_prefer = False
    for r in cycles_rle:
        # gather windows of length 4 with all distinct colors
        for i in range(len(r)-3):
            win = r[i:i+4]
            if len(set(win)) != 4:
                continue
            # canonicalize by rotation: pick lexicographically minimal rotation as key
            rots = all_rotations(win)
            key = min(tuple(rt) for rt in rots)
            counts[key] += 1
            if prefer and any(rt == prefer for rt in rots):
                seen_prefer = True
    if prefer and seen_prefer:
        return prefer
    if not counts:
        return None
    # choose the most frequent key; output it as list
    best = counts.most_common(1)[0][0]
    return list(best)

def rotate_to_match_order(rle_colors: List[str], order4: List[str]) -> Tuple[List[str], Optional[int]]:
    """
    If rle_colors contains the 4 colors in the same cyclic order (any rotation),
    rotate so that the first 4 entries align to order4. Return (rotated_list, rotation_index)
    or (original, None) if not matched.
    """
    if not rle_colors or not order4:
        return rle_colors, None
    # try all rotations of rle_colors and check the first 4 steps
    for r in range(len(rle_colors)):
        rl = rle_colors[r:] + rle_colors[:r]
        if len(rl) >= 4:
            first4 = rl[:4]
            # ensure 4 distinct in first4
            if len(set(first4)) == 4:
                # does first4 match order4 up to rotation?
                rot = rotation_index_to_match(first4, order4)
                if rot is not None:
                    # additionally, ensure the rest of rl doesn't immediately break the cycle (optional)
                    return rl, r
    return rle_colors, None

def find_flash_positions(tokens: List[str]) -> List[int]:
    """Indices where token ends with ✦."""
    idxs = []
    for i, t in enumerate(tokens):
        if t.endswith("✦"):
            idxs.append(i)
    return idxs

def color_at_index(tokens: List[str], i: int) -> Optional[str]:
    if i < 0 or i >= len(tokens):
        return None
    return strip_all(tokens[i])

def orientation_of_token(tok: str) -> Optional[str]:
    t = tok[:-1] if tok.endswith("✦") else tok
    if t.endswith("↑") or t.endswith("↓"):
        return t[-1]
    return None

def first_orientation(tokens: List[str]) -> Optional[str]:
    return orientation_of_token(tokens[0]) if tokens else None

def maybe_base4(rot_rle: List[str], order4: List[str]) -> Optional[str]:
    """
    Map colors in rot_rle to 0..3 using order4 as [0,1,2,3] if all colors are in the set and appear in that cyclic order.
    Otherwise, return None. We do not enforce exact length == 4; we just encode the list using that mapping.
    """
    if not order4 or not rot_rle:
        return None
    mapping = {order4[i]: str(i) for i in range(4)}
    for c in rot_rle:
        if c not in mapping:
            return None
    return "".join(mapping[c] for c in rot_rle)

def main():
    ap = argparse.ArgumentParser(description="Prepare per-cycle decoding tables from sequences")
    ap.add_argument("--sequences", required=True, help="Folder with *.sequence.json")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--out-json", required=True, help="Output JSON summary path")
    ap.add_argument("--prefer-order", default="", help='Optional fixed 4-color order, e.g. "Green,Yellow,Pink/Magenta,Blue"')
    args = ap.parse_args()

    seq_dir = Path(args.sequences)
    files = sorted(seq_dir.glob("*.sequence.json"))
    if not files:
        raise SystemExit("No *.sequence.json found.")

    # Load all cycles and build learning set for color order (RLE + strip extras)
    cycles_for_learning: List[List[str]] = []
    all_raw_cycles: List[Tuple[Path, int, List[str]]] = []  # (file, idx, tokens)
    for p in files:
        data = load_json(p)
        cycles = data.get("cycles") or [data.get("symbol_seq_enhanced","")]
        for idx, cyc in enumerate(cycles):
            toks_raw = toks(cyc)
            all_raw_cycles.append((p, idx, toks_raw))
            base = [strip_all(t) for t in toks_raw]
            base_rle = rle_collapse(base)
            cycles_for_learning.append(base_rle)

    prefer = [s.strip() for s in args.prefer_order.split(",")] if args.prefer_order.strip() else None
    order4 = best_four_color_order(cycles_for_learning, prefer=prefer)

    # Prepare CSV rows and summary stats
    out_rows: List[Dict[str, Any]] = []
    flash_pos_hist = Counter()
    flash_color_hist = Counter()
    start_ori_hist = Counter()
    rotate_hit = 0
    total_cycles = 0

    for p, idx, toks_raw in all_raw_cycles:
        total_cycles += 1
        base = [strip_all(t) for t in toks_raw]
        base_rle = rle_collapse(base)

        # rotation to canonical order (when possible)
        if order4:
            rot_rle, rot_idx = rotate_to_match_order(base_rle, order4)
        else:
            rot_rle, rot_idx = (base_rle, None)
        if rot_idx is not None:
            rotate_hit += 1

        # flash positions and color at those positions
        flashes = find_flash_positions(toks_raw)
        L = len(toks_raw)
        flashes_norm = [ round((i / max(1, L-1)), 3) for i in flashes ]
        flash_colors = [ color_at_index(toks_raw, i) for i in flashes ]

        for fn in flashes_norm:
            flash_pos_hist[fn] += 1
        for fc in flash_colors:
            if fc:
                flash_color_hist[fc] += 1

        # start orientation
        ori0 = first_orientation(toks_raw)
        if ori0:
            start_ori_hist[ori0] += 1

        # base-4 (if applicable)
        base4 = maybe_base4(rot_rle, order4) if order4 else None

        out_rows.append({
            "json_file": str(p.name),
            "cycle_index": idx,
            "n_tokens": L,
            "rle_colors": "|".join(base_rle),
            "rotated_to_order": "|".join(rot_rle),
            "rotation_applied": (rot_idx is not None),
            "start_orientation": ori0 or "",
            "flash_count": len(flashes),
            "flash_positions_norm": "|".join(str(x) for x in flashes_norm),
            "flash_color_hits": "|".join(fc for fc in flash_colors if fc),
            "base4_seq": base4 or "",
        })

    # Write CSV
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Summaries
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "learned_order4": order4 or [],
        "cycles_total": total_cycles,
        "cycles_rotated": rotate_hit,
        "start_orientation_hist": dict(start_ori_hist),
        "flash_pos_hist_norm": dict(flash_pos_hist),
        "flash_color_hist": dict(flash_color_hist),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_csv} and {out_json}")
    if order4:
        print("[INFO] learned 4-color cyclic order:", " → ".join(order4))
        print(f"[INFO] rotated {rotate_hit}/{total_cycles} cycles to that order")

if __name__ == "__main__":
    main()
