#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_tokens.py  —  post-processor with intra-purity gate

Merges duplicate/confusable tokens across:
  - symbol_codebook.json
  - *.sequence.json (in a folder)

Sources decisions from:
  - Gemini audit (audit_report.json): inter similarity (prototypes) and intra verdicts
  - Optional manual map (--map "D=H,B=E")

Auto-merge requires BOTH:
  1) inter similarity >= --threshold
  2) each token meets intra-purity gate:
       purity >= --purity-min  AND  verdict_count >= --min-intra-samples
     (purity can optionally count only verdicts with similarity >= --purity-sim-min)

Manual merges are applied first and bypass purity by default (use --gate-manual to gate them too).

Outputs:
  --out/symbol_codebook.json
  --out/*.sequence.json
  --out/merge_log.json

No third-party deps required (standard library only).
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

# ---------- IO helpers ----------

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- CLI map parsing ----------

def parse_manual_map(s: str):
    """
    "D=H,B=E" -> {"D":"H","B":"E"}  (left merges into right)
    """
    mapping = {}
    if not s:
        return mapping
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad --map fragment: {part}. Use like D=H,B=E")
        src, dst = [x.strip() for x in part.split("=", 1)]
        if not src or not dst:
            raise ValueError(f"Bad --map pair: {part}")
        mapping[src] = dst
    return mapping

# ---------- Codebook readers ----------

def load_codebook(cb_path: Path):
    """
    Supports two shapes:
      A) {"clusters":[{"token": "A", "count": ..., "prototype": {...}}, ...]}
      B) {"legend": {"A": {..., "count": ..., "prototype": {...}}, ...}}
    Returns (cb_object, clusters_dict, model_tag, counts_dict)
    """
    cb = load_json(cb_path)
    clusters = {}
    model = None

    if isinstance(cb.get("clusters"), list):
        model = "clusters"
        for c in cb["clusters"]:
            tok = c.get("token")
            if tok:
                clusters[tok] = c
    elif isinstance(cb.get("legend"), dict):
        model = "legend"
        for tok, meta in cb["legend"].items():
            clusters[tok] = meta
    else:
        raise ValueError("Unrecognized codebook format (expected 'clusters' list or 'legend' dict)")

    counts = {}
    for tok, meta in clusters.items():
        cnt = meta.get("count")
        counts[tok] = int(cnt) if isinstance(cnt, int) else 0

    return cb, clusters, model, counts

# ---------- Audit readers ----------

def load_inter_pairs(audit: dict):
    """
    Return list of inter pairs:
      [{"a":"A","b":"B","similarity":0.83}, ...] sorted desc by similarity.
    """
    inter = audit.get("inter", {})
    tokens = inter.get("tokens") or inter.get("Tokens") or []
    matrix = inter.get("matrix") or inter.get("similarity_matrix") or {}
    pairs = inter.get("top_confusable_pairs") or []

    # normalize pairs
    norm_pairs = []
    for p in pairs:
        a = p.get("a") or p.get("A") or p.get("t1")
        b = p.get("b") or p.get("B") or p.get("t2")
        sim = p.get("similarity") or p.get("score") or 0.0
        if a and b:
            try:
                norm_pairs.append({"a": str(a), "b": str(b), "similarity": float(sim)})
            except Exception:
                continue

    # if empty, derive from matrix
    if not norm_pairs and isinstance(matrix, dict):
        toks = tokens or list(matrix.keys())
        for i in range(len(toks)):
            for j in range(i+1, len(toks)):
                ti, tj = toks[i], toks[j]
                try:
                    sim = float(matrix.get(ti, {}).get(tj, None))
                except Exception:
                    sim = None
                if sim is not None:
                    norm_pairs.append({"a": ti, "b": tj, "similarity": sim})

    norm_pairs.sort(key=lambda d: d["similarity"], reverse=True)
    return norm_pairs

def load_intra_purity(audit: dict, sim_min: float | None, min_samples: int):
    """
    Compute per-token purity from audit["intra"].
    Returns: dict token -> {"purity": float, "n": int}
    purity = (# verdicts same_symbol==True AND (similarity>=sim_min if provided)) / (# verdicts counted)
    If no entries or n < min_samples, token will not pass the gate.
    """
    intra = audit.get("intra", {}) or {}
    purity = {}
    for tok, block in intra.items():
        verdicts = block.get("verdicts") or []
        num = 0
        den = 0
        for v in verdicts:
            same = bool(v.get("same_symbol", False))
            sim  = v.get("similarity", None)
            if sim_min is not None:
                try:
                    if sim is None or float(sim) < float(sim_min):
                        # don't count very low-sim "agreements" toward purity
                        continue
                except Exception:
                    continue
            den += 1
            if same:
                num += 1
        purity[tok] = {"purity": (num/den) if den > 0 else 0.0, "n": den}
    return purity

# ---------- Merge planner (with gates) ----------

def plan_merges(cb_counts: dict,
                inter_pairs: list,
                manual_map: dict,
                threshold: float,
                purity: dict,
                purity_min: float,
                min_intra_samples: int,
                gate_manual: bool):
    """
    Build final mapping loser->winner.

    Order:
      1) Apply manual_map first (optionally gate them if gate_manual).
      2) Auto merges from inter_pairs where:
           - sim >= threshold
           - BOTH tokens pass intra gate: purity>=purity_min AND n>=min_intra_samples
         Winner chosen by larger count (then lexicographic).

    Returns mapping and a verbose 'reasons' log per candidate pair.
    """
    mapping = {}
    reasons = []

    def final(tok):
        seen = set()
        while tok in mapping and tok not in seen:
            seen.add(tok)
            tok = mapping[tok]
        return tok

    # 1) manual first
    for s, d in (manual_map or {}).items():
        if s == d:
            continue
        # optional gating for manual merges
        if gate_manual:
            ps = purity.get(s, {"purity": 0.0, "n": 0})
            pd = purity.get(d, {"purity": 0.0, "n": 0})
            gate_ok = (ps["purity"] >= purity_min and ps["n"] >= min_intra_samples and
                       pd["purity"] >= purity_min and pd["n"] >= min_intra_samples)
            reasons.append({"pair": f"{s}={d}", "type": "manual", "applied": gate_ok,
                            "reason": f"gate_manual; {s} purity={ps}, {d} purity={pd}"})
            if not gate_ok:
                continue

        mapping[s] = d
        reasons.append({"pair": f"{s}={d}", "type": "manual", "applied": True, "reason": "manual map"})

    # 2) auto from inter pairs
    for p in inter_pairs:
        s1, s2, sim = p["a"], p["b"], float(p["similarity"])
        if sim < threshold:
            reasons.append({"pair": f"{s1}~{s2}", "type": "auto", "applied": False,
                            "reason": f"sim {sim:.3f} < threshold {threshold:.3f}"})
            continue

        f1, f2 = final(s1), final(s2)
        if f1 == f2:
            reasons.append({"pair": f"{s1}->{f1} vs {s2}->{f2}", "type": "auto", "applied": False,
                            "reason": "already same after prior merges"})
            continue

        ps = purity.get(f1, {"purity": 0.0, "n": 0})
        pd = purity.get(f2, {"purity": 0.0, "n": 0})

        # intra purity gate
        ok1 = ps["purity"] >= purity_min and ps["n"] >= min_intra_samples
        ok2 = pd["purity"] >= purity_min and pd["n"] >= min_intra_samples
        if not (ok1 and ok2):
            reasons.append({"pair": f"{f1}~{f2}", "type": "auto", "applied": False,
                            "reason": f"purity gate failed: {f1} {ps}, {f2} {pd}"})
            continue

        # choose winner by larger count (fallback token name)
        c1, c2 = cb_counts.get(f1, 0), cb_counts.get(f2, 0)
        if (c1 > c2) or (c1 == c2 and str(f1) < str(f2)):
            winner, loser = f1, f2
        else:
            winner, loser = f2, f1

        mapping[loser] = winner
        reasons.append({"pair": f"{loser}->{winner}", "type": "auto", "applied": True,
                        "reason": f"sim {sim:.3f} >= {threshold:.3f} AND purity gates passed"})

    # compress transitive chains
    collapsed = {}
    for s in list(mapping.keys()):
        d = mapping[s]
        seen = set()
        while d in mapping and d not in seen:
            seen.add(d); d = mapping[d]
        collapsed[s] = d

    return collapsed, reasons

# ---------- Apply mapping ----------

def apply_mapping_to_codebook(cb, model, mapping: dict):
    """
    Return updated codebook and set of present tokens after merges.
    """
    if not mapping:
        return cb, set()

    def final(tok):
        while tok in mapping:
            tok = mapping[tok]
        return tok

    present = set()

    if model == "clusters":
        merged = {}
        for c in cb["clusters"]:
            tok = c.get("token")
            if not tok:
                continue
            new_tok = final(tok)
            c["token"] = new_tok
            if new_tok not in merged:
                merged[new_tok] = c
            else:
                # sum counts if present
                try:
                    merged[new_tok]["count"] = int(merged[new_tok].get("count", 0)) + int(c.get("count", 0))
                except Exception:
                    pass
        cb["clusters"] = list(merged.values())
        present = {c["token"] for c in cb["clusters"]}

    elif model == "legend":
        merged = {}
        for tok, meta in cb["legend"].items():
            new_tok = final(tok)
            if new_tok not in merged:
                merged[new_tok] = meta
                merged[new_tok]["token"] = new_tok
            else:
                try:
                    merged[new_tok]["count"] = int(merged[new_tok].get("count", 0)) + int(meta.get("count", 0))
                except Exception:
                    pass
        cb["legend"] = merged
        present = set(cb["legend"].keys())

    if "n_clusters" in cb:
        try:
            cb["n_clusters"] = len(present)
        except Exception:
            pass

    return cb, present

def update_symbol_seq(seq_obj: dict, mapping: dict) -> None:
    """
    Update per-sequence JSON in-place:
      - legend keys
      - events[].token if present
      - symbol_seq (best-effort: from events if present, else 1-char translate)
    """
    if not mapping:
        return

    def final(tok):
        while tok in mapping:
            tok = mapping[tok]
        return tok

    # legend
    if isinstance(seq_obj.get("legend"), dict):
        new_leg = {}
        for tok, meta in seq_obj["legend"].items():
            new_tok = final(tok)
            if new_tok not in new_leg:
                new_leg[new_tok] = meta
                if isinstance(new_leg[new_tok], dict):
                    new_leg[new_tok]["token"] = new_tok
            else:
                try:
                    new_leg[new_tok]["count"] = int(new_leg[new_tok].get("count",0)) + int(meta.get("count",0))
                except Exception:
                    pass
        seq_obj["legend"] = new_leg

    # events
    if isinstance(seq_obj.get("events"), list):
        for ev in seq_obj["events"]:
            if isinstance(ev.get("token"), str):
                ev["token"] = final(ev["token"])
        # rebuild symbol_seq from events if timestamps/indices exist
        try:
            evs = seq_obj["events"]
            def k(ev):
                st = ev.get("start_ts"); idx = ev.get("event_index")
                stv = float(st) if isinstance(st,(int,float)) else 1e18
                idxv = int(idx) if isinstance(idx,int) else 1e9
                return (stv, idxv)
            evs_sorted = sorted(evs, key=k)
            seq_obj["symbol_seq"] = "".join(e.get("token","") for e in evs_sorted)
        except Exception:
            pass

    # symbol_seq (char translate if single-char tokens)
    sym = seq_obj.get("symbol_seq")
    if isinstance(sym, str) and sym and isinstance(seq_obj.get("legend"), dict):
        keys = list(seq_obj["legend"].keys())
        if keys and all(isinstance(k, str) and len(k) == 1 for k in keys):
            trans = {ord(s): ord(final(s)) for s in mapping.keys() if isinstance(s,str) and len(s)==1 and isinstance(final(s),str) and len(final(s))==1}
            seq_obj["symbol_seq"] = sym.translate(trans)

def apply_mapping_to_sequences(sequences_dir: Path, out_dir: Path, mapping: dict):
    changed = []
    for p in sequences_dir.glob("*.sequence.json"):
        try:
            obj = load_json(p)
        except Exception:
            continue
        update_symbol_seq(obj, mapping)
        out_p = out_dir / p.name
        save_json(out_p, obj)
        changed.append(p.name)
    return changed

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Merge duplicate/confusable tokens with intra-purity gate.")
    ap.add_argument("--codebook", required=True, help="Path to symbol_codebook.json")
    ap.add_argument("--sequences", required=True, help="Path to folder with *.sequence.json")
    ap.add_argument("--audit", required=True, help="Path to audit_report.json")
    ap.add_argument("--out", required=True, help="Output directory for merged results")

    ap.add_argument("--threshold", type=float, default=0.80, help="Inter similarity threshold for auto-merge")
    ap.add_argument("--purity-min", type=float, default=0.80, help="Intra purity gate (fraction of 'same_symbol' votes)")
    ap.add_argument("--min-intra-samples", type=int, default=8, help="Min verdicts per token to trust purity")
    ap.add_argument("--purity-sim-min", type=float, default=0.60, help="Only count intra verdicts with similarity≥this; set <0 to ignore similarity")
    ap.add_argument("--map", type=str, default="", help='Manual merges, e.g., "D=H,B=E"')
    ap.add_argument("--gate-manual", action="store_true", help="Also gate manual merges by purity")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; write merge_log.json but do not modify files")
    args = ap.parse_args()

    cb_path = Path(args.codebook)
    seq_dir = Path(args.sequences)
    audit_path = Path(args.audit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    codebook, clusters, model, counts = load_codebook(cb_path)
    audit = load_json(audit_path)
    inter_pairs = load_inter_pairs(audit)
    sim_min = float(args.purity_sim_min)
    sim_min = None if sim_min < 0 else sim_min
    purity = load_intra_purity(audit, sim_min=sim_min, min_samples=int(args.min_intra_samples))
    manual_map = parse_manual_map(args.map)

    # Plan merges (mapping + reasons)
    mapping, reasons = plan_merges(
        cb_counts=counts,
        inter_pairs=inter_pairs,
        manual_map=manual_map,
        threshold=float(args.threshold),
        purity=purity,
        purity_min=float(args.purity_min),
        min_intra_samples=int(args.min_intra_samples),
        gate_manual=bool(args.gate_manual)
    )

    # Write plan log
    plan_log = {
        "threshold": float(args.threshold),
        "purity_min": float(args.purity_min),
        "min_intra_samples": int(args.min_intra_samples),
        "purity_sim_min": sim_min if sim_min is not None else "ignored",
        "manual_map": manual_map,
        "mapping_final": mapping,
        "reasons": reasons,
        "purity_table": purity,
        "top_inter_pairs": inter_pairs[:20]
    }
    save_json(out_dir / "merge_log.json", plan_log)
    print("[plan] mapping to apply:", json.dumps(mapping, indent=2))

    if args.dry_run:
        print("[dry-run] plan saved; no files written.")
        return

    # Apply mapping
    new_cb, present = apply_mapping_to_codebook(codebook, model, mapping)
    save_json(out_dir / "symbol_codebook.json", new_cb)
    files_changed = apply_mapping_to_sequences(seq_dir, out_dir, mapping)

    print(f"[ok] codebook → {out_dir / 'symbol_codebook.json'}")
    print(f"[ok] sequences updated → {len(files_changed)} files")
    print(f"[done] merge_log → {out_dir / 'merge_log.json'}")

if __name__ == "__main__":
    main()
