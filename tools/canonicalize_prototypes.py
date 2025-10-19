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

def apply_human_actions(codebook: dict, actions_path: Path) -> tuple[dict, Dict[int, int]]:
    actions = load_json(actions_path)

    clusters = codebook.get("clusters", [])
    by_id = {c.get("cluster_id"): c for c in clusters if c.get("cluster_id") is not None}

    id_remap: Dict[int, int] = {}
    for merge in actions.get("merges", []):
        tgt = merge.get("target_id")
        if tgt is None:
            continue
        for src in merge.get("source_ids", []):
            if src == tgt:
                continue
            id_remap[src] = tgt
            if src in by_id and tgt in by_id:
                tgt_cluster = by_id[tgt]
                src_cluster = by_id[src]
                try:
                    tgt_count = int(tgt_cluster.get("count", 0))
                except Exception:
                    tgt_count = 0
                try:
                    src_count = int(src_cluster.get("count", 0))
                except Exception:
                    src_count = 0
                tgt_cluster["count"] = tgt_count + src_count

    keep = []
    for c in clusters:
        cid = c.get("cluster_id")
        if cid in id_remap:
            continue
        keep.append(c)
    codebook["clusters"] = keep
    by_id = {c.get("cluster_id"): c for c in keep if c.get("cluster_id") is not None}

    for rename in actions.get("renames", []):
        cid = rename.get("cluster_id")
        new_name = (rename.get("new_name") or "").strip()
        if not new_name:
            continue
        if cid in by_id:
            by_id[cid]["name"] = new_name

    if id_remap:
        codebook["cluster_id_remap"] = id_remap

    return codebook, id_remap

def remap_sequences(seq_dir: Path, id_remap: Dict[int, int]) -> None:
    if not id_remap:
        return
    seq_paths = sorted(seq_dir.glob("*.sequence.json"))
    for seq_path in seq_paths:
        try:
            doc = json.loads(seq_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        changed = False
        for event in doc.get("events", []):
            cid = event.get("cluster_id")
            if cid in id_remap:
                event["cluster_id"] = id_remap[cid]
                changed = True
        if changed:
            seq_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Canonicalize visually similar prototypes via perceptual hash.")
    ap.add_argument("--codebook", required=True, help="Path to symbol_codebook.json")
    ap.add_argument("--thumbs",   required=True, help="Path to thumbs_obj folder")
    ap.add_argument("--out",      required=True, help="Output JSON path (non-destructive)")
    ap.add_argument("--method",   default="ahash", choices=["ahash","phash","dhash"], help="Perceptual hash method")
    ap.add_argument("--threshold", type=int, default=8, help="Max Hamming distance to group")
    ap.add_argument("--human-actions", type=Path, help="Path to human_review_actions.json (merges/renames/flags)")
    ap.add_argument(
        "--sequences",
        type=Path,
        help="Directory containing *.sequence.json files to optionally rewrite",
    )
    ap.add_argument(
        "--remap-sequences",
        action="store_true",
        help="Apply cluster_id remap to sequences (requires --human-actions)",
    )
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

    id_remap: Dict[int, int] = {}
    actions_path = getattr(args, "human_actions", None)
    if actions_path:
        actions_path = Path(actions_path)
        if actions_path.exists():
            codebook, id_remap = apply_human_actions(codebook, actions_path)
        else:
            print(f"[WARN] human actions file not found: {actions_path}")

    save_json(Path(args.out), codebook)
    print(f"[OK] Wrote canonicalized codebook to {args.out}")

    if getattr(args, "remap_sequences", False) and id_remap:
        seq_dir = getattr(args, "sequences", None)
        if not seq_dir:
            print("[WARN] --remap-sequences requested but --sequences not provided; skipping remap.")
        else:
            remap_sequences(Path(seq_dir), id_remap)
            print(f"[OK] Remapped sequences in {seq_dir}")

if __name__ == "__main__":
    main()
