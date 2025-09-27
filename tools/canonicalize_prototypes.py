#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canonicalize_prototypes.py
Group visually near-duplicate cluster prototypes via perceptual hashing and
write canonical names back into the codebook (non-destructive copy by default).

Usage:
  python canonicalize_prototypes.py ^
    --codebook "C:\...\public\sequences\symbol_codebook.json" ^
    --thumbs   "C:\...\public\thumbs_obj" ^
    --out      "C:\...\public\sequences\symbol_codebook_canon.json" ^
    --method   "ahash" ^
    --threshold 8

Notes:
- Uses imagehash (Pillow) for pHashing; threshold is Hamming distance.
- Works on your current codebook["clusters"] (token-level prototypes).
- If you already have names, we preserve them and assign that as canonical
  for the group; otherwise we use the first token as a stable label.
"""

import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import imagehash

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def compute_hash(img_path: Path, method: str) -> imagehash.ImageHash | None:
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        return None
    m = method.lower()
    if m == "phash":
        return imagehash.phash(im)
    if m == "dhash":
        return imagehash.dhash(im)
    # default average
    return imagehash.average_hash(im)

def group_by_hash(protos: List[dict], thumbs_root: Path, method: str, thr: int) -> List[List[dict]]:
    """
    O(n^2) grouping by Hamming distance <= thr.
    Each proto item: {"cluster_id":..., "token":..., "thumb": Path or None}
    """
    hashes: Dict[int, imagehash.ImageHash] = {}
    items: List[Tuple[int, dict]] = []   # (index, proto dict)
    for i, p in enumerate(protos):
        thumb = p.get("prototype", {}).get("thumb_obj", "")
        img = thumbs_root / thumb if thumb else None
        h = compute_hash(img, method) if (img and img.exists()) else None
        hashes[i] = h
        items.append((i, p))

    used = set()
    groups: List[List[dict]] = []
    for i, pi in items:
        if i in used:
            continue
        hi = hashes[i]
        grp = [pi]
        used.add(i)
        if hi is None:
            groups.append(grp)
            continue
        for j, pj in items:
            if j in used or j == i:
                continue
            hj = hashes[j]
            if hj is None:
                continue
            if (hi - hj) <= thr:
                grp.append(pj)
                used.add(j)
        groups.append(grp)
    return groups

def pick_canonical_name(group: List[dict]) -> str:
    """
    If any proto already has a 'name', reuse the most common name;
    else use the first token as a canonical placeholder (e.g., 'A', 'B', 'AA').
    """
    # In your current codebook, 'name' may not be present; this is future-proof.
    names = [str(p.get("name","")).strip() for p in group if p.get("name")]
    if names:
        # Most frequent non-empty
        best = sorted(((n, names.count(n)) for n in set(names)), key=lambda t: (-t[1], t[0]))[0][0]
        return best
    # Fallback to token
    toks = [p.get("token","") for p in group if p.get("token")]
    return toks[0] if toks else "canon_proto"

def main():
    ap = argparse.ArgumentParser(description="Canonicalize visually similar prototypes via perceptual hash.")
    ap.add_argument("--codebook", required=True, help="Path to symbol_codebook.json")
    ap.add_argument("--thumbs",   required=True, help="Path to thumbs_obj folder")
    ap.add_argument("--out",      required=True, help="Output JSON path (non-destructive)")
    ap.add_argument("--method",   default="ahash", choices=["ahash","phash","dhash"], help="Perceptual hash method")
    ap.add_argument("--threshold", type=int, default=8, help="Max Hamming distance to group")
    args = ap.parse_args()

    codebook = load_json(Path(args.codebook))
    thumbs = Path(args.thumbs)

    protos = codebook.get("clusters", [])
    if not protos:
        print("[WARN] codebook.clusters is empty; nothing to canonicalize.")
        save_json(Path(args.out), codebook)
        return

    groups = group_by_hash(protos, thumbs, args.method, int(args.threshold))
    # Annotate each cluster with canonical_name and group_id
    for gid, grp in enumerate(groups, start=1):
        canon = pick_canonical_name(grp)
        for p in grp:
            p["canonical_name"] = canon
            p["canonical_group_id"] = gid

    # Optionally: build a registry dump
    codebook["canonical_registry"] = [
        {
            "group_id": gid,
            "canonical_name": pick_canonical_name(grp),
            "members": [ {"cluster_id": int(p.get("cluster_id",-1)), "token": p.get("token",""),
                          "thumb_obj": p.get("prototype",{}).get("thumb_obj","") } for p in grp ]
        }
        for gid, grp in enumerate(groups, start=1)
    ]

    save_json(Path(args.out), codebook)
    print(f"[OK] Wrote canonicalized codebook to {args.out}")

if __name__ == "__main__":
    main()
