#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_report_flagging.py
Create a human-review queue from the Gemini cluster audit + codebook.

Flags two classes:
  1) Naming inconsistencies between visually confusable pairs (high similarity)
     where both members already have names (and the names differ).
  2) High-frequency unknowns (no name but large window counts).

Usage:
  python audit_report_flagging.py ^
    --audit    "C:\...\public\audit\audit_report.json" ^
    --codebook "C:\...\public\sequences\symbol_codebook.json" ^
    --out      "C:\...\public\sequences\human_review_queue.json" ^
    --sim-th   0.70 ^
    --min-windows 10
"""

import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple

def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj: dict) -> None:
    # ensure parent directory exists before writing
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def make_by_cluster(codebook: dict) -> Dict[int, dict]:
    bycid = {}
    for c in codebook.get("clusters", []):
        cid = int(c.get("cluster_id", -1))
        bycid[cid] = c
    return bycid

def name_of(proto: dict) -> str:
    # name may be absent in your current codebook; support future integration w/ Gemini naming
    nm = (proto.get("name") or proto.get("canonical_name") or "").strip()
    return nm

def main():
    ap = argparse.ArgumentParser(description="Build a human review queue from audit + codebook")
    ap.add_argument("--audit", required=True, help="Path to audit_report.json")
    ap.add_argument("--codebook", required=True, help="Path to symbol_codebook.json")
    ap.add_argument("--out", required=True, help="Output JSON path for human_review_queue.json")
    ap.add_argument("--sim-th", type=float, default=0.70, help="Inter similarity threshold to flag")
    ap.add_argument("--min-windows", type=int, default=10, help="Minimum window count to flag unnamed frequent prototypes")
    args = ap.parse_args()

    audit = load_json(Path(args.audit))
    codebook = load_json(Path(args.codebook))
    bycid = make_by_cluster(codebook)

    # Build window counts from sequences if present in codebook (optional future expansion).
    # For now, use per-cluster "count" as the proxy (how many events in that cluster).
    counts = { int(c.get("cluster_id",-1)) : int(c.get("count",0)) for c in codebook.get("clusters",[]) }

    review_queue: List[dict] = []

    # 1) Flag naming inconsistencies among confusable pairs (use pairs_detailed if available)
    inter = audit.get("inter", {})
    detailed_pairs = inter.get("pairs_detailed", [])
    for pair in detailed_pairs:
        cid_a = int(pair.get("cluster_id_a", -1))
        cid_b = int(pair.get("cluster_id_b", -1))
        sim   = float(pair.get("similarity", 0.0))
        if sim < float(args.sim_th):
            continue
        pa = bycid.get(cid_a, {})
        pb = bycid.get(cid_b, {})
        nm_a = name_of(pa)
        nm_b = name_of(pb)
        if nm_a and nm_b and (nm_a.lower() != nm_b.lower()):
            review_queue.append({
                "type": "naming_inconsistency",
                "similarity": sim,
                "a": {"cluster_id": cid_a, "token": pa.get("token",""), "name": nm_a,
                      "thumb_obj": pa.get("prototype",{}).get("thumb_obj","")},
                "b": {"cluster_id": cid_b, "token": pb.get("token",""), "name": nm_b,
                      "thumb_obj": pb.get("prototype",{}).get("thumb_obj","")},
                "reason": f"High inter-similarity but different names ('{nm_a}' vs '{nm_b}')"
            })
    # Fallback: derive pairs from inter.matrix if pairs_detailed is missing
        if not detailed_pairs:
            tokens = inter.get("tokens", [])
            matrix = inter.get("matrix", {})
            # map token -> cluster_id using the codebook clusters
            tok2cid = { str(c.get("token","")): int(c.get("cluster_id",-1)) for c in codebook.get("clusters", []) }
            for i in range(len(tokens)):
                ti = tokens[i]
                for j in range(i+1, len(tokens)):
                    tj = tokens[j]
                    try:
                        sim = float(matrix.get(ti, {}).get(tj, None))
                    except Exception:
                        sim = None
                    if sim is None:
                        continue
                    detailed_pairs.append({
                        "token_a": ti, "cluster_id_a": tok2cid.get(ti, -1),
                        "token_b": tj, "cluster_id_b": tok2cid.get(tj, -1),
                        "similarity": sim
                    })

    # 2) High-frequency unknowns (good candidates to be named)
    for cid, c in bycid.items():
        nm = name_of(c)
        if nm:
            continue
        if counts.get(cid, 0) >= int(args.min_windows):
            review_queue.append({
                "type": "new_name_candidate",
                "cluster_id": cid,
                "token": c.get("token",""),
                "thumb_obj": c.get("prototype",{}).get("thumb_obj",""),
                "windows": counts[cid],
                "reason": "High-frequency cluster lacks name"
            })

    save_json(Path(args.out), {"review_queue": review_queue})
    print(f"[OK] wrote {args.out} with {len(review_queue)} review items")

if __name__ == "__main__":
    main()
