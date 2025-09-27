#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gemini_cluster_audit.py

Use Gemini to visually audit whether build_sequences.py produced sensible clusters:
- Intra-cluster: prototype + random samples → same/different, similarity, color/shape deltas.
- Inter-cluster: prototype-only similarity matrix (how confusable are tokens).

Inputs:
  --codebook  path/to/symbol_codebook.json
  --sequences dir with *.sequence.json (contains events w/ cluster_id + thumb_obj)
  --thumbs    dir with object crops (thumbs_obj)
  --out       output folder for audit_report.json/html

Env:
  GOOGLE_API_KEY must be set (or .env with GOOGLE_API_KEY=...)

Model:
  gemini-2.5-flash (multimodal, supports images)

pip:
  pip install google-genai python-dotenv opencv-python pillow
"""

from __future__ import annotations
import os, sys, json, random, base64, argparse, textwrap
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"

# ---------------------- I/O helpers ----------------------

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"): return "image/jpeg"
    if ext == ".png": return "image/png"
    if ext == ".webp": return "image/webp"
    return "image/jpeg"

def image_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        b = f.read()
    return "data:{};base64,{}".format(guess_mime(path), base64.b64encode(b).decode("ascii"))

# ---------------------- Gemini response helpers ----------------------

def extract_text(resp) -> str:
    # robustly pull text; fallback to parts
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    out = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if content and getattr(content, "parts", None):
            for p in content.parts:
                txt = getattr(p, "text", None)
                if isinstance(txt, str) and txt:
                    out.append(txt)
    return "".join(out)

def json_only_or_raise(raw: str) -> dict:
    s = raw.strip()
    # strip ``` fences if any
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[len("json"):].lstrip()
    # remove trailing fences if present
    s = s.strip().strip("`").strip()
    try:
        return json.loads(s)
    except Exception as e:
        raise RuntimeError(f"Gemini did not return parseable JSON.\n---RAW START---\n{raw}\n---RAW END---") from e

# ---------------------- Data structures ----------------------

@dataclass
class Cluster:
    cluster_id: int
    token: str
    count: int
    prototype_thumb: str

# ---------------------- Prompt templates ----------------------

INTRA_PROMPT = textwrap.dedent("""
You are auditing visual clusters of tiny object crops.

Task: For the images provided in order, decide whether each SAMPLE depicts the **same symbol** as the PROTOTYPE.
Definition of "same symbol":
- Prioritize **shape silhouette + internal structure** and **dominant color/hue**.
- Ignore translation/scale/crop, small blur/lighting, minor background artifacts, and watermark/overlay text.
- Treat different hues (distinct color families) as different symbols unless the change is clearly a lighting artifact.

Rubric weights:
- color/hue 0.45, shape 0.45, texture/markings 0.10

INPUT ORDER:
- First image = PROTOTYPE
- Next N images = SAMPLES [S1..SN] in the same order as listed.

Return ONLY a single JSON object with exactly:
{
  "prototype_name": "<string>",
  "verdicts": [
    {
      "sample_name": "<string>",
      "same_symbol": true|false,
      "similarity": 0.0-1.0,   // overall similarity score using the rubric
      "color_delta": "small|moderate|large",
      "shape_delta": "small|moderate|large",
      "notes": "<<=12 words>"
    },
    ...
  ],
  "summary": "<=1 sentence overall impression>"
}
Do not include any additional keys. Do not add prose outside JSON.
""").strip()

INTER_PROMPT = textwrap.dedent("""
You are auditing whether **cluster prototypes** are visually confusable.

Task: Given K prototype images (tokens), produce a symmetric similarity matrix using the rubric:
- color/hue 0.45, shape 0.45, texture 0.10
- Scores in [0,1], with 1.0 = identical symbol, ~0.0 = clearly different.

Rules:
- Use **circular hue** reasoning; large hue differences => lower similarity unless obviously lighting-only.
- Ignore background/watermark text; focus on the object.

INPUT ORDER:
- We provide a list of tokens [T1..Tk] in the exact order the images are attached.

Return ONLY a single JSON object:
{
  "tokens": ["T1","T2",...,"Tk"],
  "similarity_matrix": { "T1": {"T1":1.0,"T2":0.x,...}, "T2": {...}, ... },
  "top_confusable_pairs": [ {"a":"Ti","b":"Tj","similarity":0.x}, ... ]   // 3-10 items, sorted desc by similarity, omit pairs < 0.3
}
No extra keys. No prose outside JSON.
""").strip()

# ---------------------- Core audit ----------------------

def collect_members(sequences_dir: Path) -> Dict[int, List[str]]:
    """Build cluster_id -> [thumb_obj names] from all *.sequence.json."""
    members: Dict[int, List[str]] = {}
    for p in sequences_dir.glob("*.sequence.json"):
        try:
            j = load_json(p)
        except Exception:
            continue
        for ev in j.get("events", []):
            cid = ev.get("cluster_id", None)
            tname = (ev.get("thumb_obj") or "").strip()
            if cid is None or not tname:
                continue
            members.setdefault(int(cid), []).append(tname)
    # de-dup while preserving order
    for k,v in list(members.items()):
        seen = set(); out=[]
        for name in v:
            if name in seen: continue
            seen.add(name); out.append(name)
        members[k] = out
    return members

def build_clusters(codebook: dict) -> List[Cluster]:
    clusters: List[Cluster] = []
    for c in codebook.get("clusters", []):
        clusters.append(Cluster(
            cluster_id=int(c.get("cluster_id", -1)),
            token=str(c.get("token", "")),
            count=int(c.get("count", 0)),
            prototype_thumb=str(c.get("prototype", {}).get("thumb_obj", ""))
        ))
    clusters.sort(key=lambda x: x.count, reverse=True)
    return clusters

def make_image_part(path: Path) -> types.Part:
    with path.open("rb") as f:
        b = f.read()
    return types.Part(inline_data=types.Blob(data=b, mime_type=guess_mime(path)))

def audit_intra_for_cluster(client, cluster: Cluster, sample_names: List[str], thumbs_dir: Path) -> dict:
    # Build content: text listing order, then prototype image, then sample images
    parts: List[types.Part] = []
    proto_name = cluster.prototype_thumb or (sample_names[0] if sample_names else "")
    listing = ["Images appear in this order:", f"1) PROTOTYPE = {proto_name}"]
    for i, s in enumerate(sample_names, start=2):
        listing.append(f"{i}) SAMPLE = {s}")
    parts.append(types.Part(text="\n".join(listing)))
    # Add images
    proto_path = thumbs_dir / proto_name
    parts.append(make_image_part(proto_path))
    for s in sample_names:
        parts.append(make_image_part(thumbs_dir / s))
    parts.append(types.Part(text=INTRA_PROMPT))

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(temperature=0.1)
    )
    raw = extract_text(resp)
    return json_only_or_raise(raw)

def audit_inter_prototypes(client, clusters: List[Cluster], thumbs_dir: Path, max_protos: int=12) -> dict:
    cl = clusters[:max_protos]
    parts: List[types.Part] = []
    tokens = [c.token for c in cl]
    parts.append(types.Part(text="Tokens (order): " + ", ".join(tokens)))
    for c in cl:
        name = c.prototype_thumb
        parts.append(make_image_part(thumbs_dir / name))
    parts.append(types.Part(text=INTER_PROMPT))
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(temperature=0.1)
    )
    raw = extract_text(resp)
    data = json_only_or_raise(raw)

    # Build pairs_detailed so downstream HIL tools know the actual cluster_ids
    detailed = []
    tokens_out = data.get("tokens", tokens)
    sim = data.get("similarity_matrix", {})
    # Map token -> cluster_id for the set we audited in-order
    tok2cid = {c.token: c.cluster_id for c in cl}
    K = len(tokens_out)
    for i in range(K):
        ti = tokens_out[i]
        for j in range(i+1, K):
            tj = tokens_out[j]
            s = None
            try:
                s = float(sim.get(ti, {}).get(tj, None))
            except Exception:
                pass
            if s is None:
                continue
            detailed.append({
                "token_a": ti, "cluster_id_a": tok2cid.get(ti, -1),
                "token_b": tj, "cluster_id_b": tok2cid.get(tj, -1),
                "similarity": s
            })
    return {
        "tokens": tokens_out,
        "matrix": sim,
        "top_confusable_pairs": data.get("top_confusable_pairs", []),
        "pairs_detailed": detailed
    }

# ---------------------- HTML report ----------------------

def render_html(audit: dict, out_html: Path, thumbs_dir: Path):
    def fmt_num(x, nd=2):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    def safe_img_b64(p: Path) -> str:
        try:
            return image_to_base64(p)
        except Exception:
            return ""

    css = """
    body{font-family:system-ui,Segoe UI,Arial;margin:24px;color:#111}
    h1{margin:0 0 8px 0}
    .cluster{border:1px solid #ddd;border-radius:12px;padding:16px;margin:16px 0}
    .row{display:flex;gap:16px;flex-wrap:wrap}
    .card{border:1px solid #eee;border-radius:10px;padding:10px;width:200px}
    .card img{width:100%;height:auto;border-radius:8px;display:block}
    .proto{border:2px solid #333}
    .meta{font-size:12px;color:#444}
    .badge{display:inline-block;padding:2px 6px;border-radius:6px;font-size:11px;margin-left:6px;background:#f2f2f2}
    table{border-collapse:collapse;margin-top:12px}
    th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px}
    """
    html = [f"<html><head><meta charset='utf-8'><style>{css}</style><title>Gemini Cluster Audit</title></head><body>"]
    html.append("<h1>Gemini Cluster Audit</h1>")
    html.append("<p>This report summarizes Gemini's visual checks of your clusters: intra-cluster agreement and inter-cluster similarity.</p>")

    # Inter summary
    if audit.get("inter"):
        html.append("<h2>Inter-cluster Similarity (Prototypes)</h2>")
        inter = audit["inter"]
        tokens = inter.get("tokens", [])
        matrix = inter.get("matrix", {})
        # table
        html.append("<table><tr><th></th>" + "".join(f"<th>{t}</th>" for t in tokens) + "</tr>")
        for ti in tokens:
            row = [f"<tr><th>{ti}</th>"]
            for tj in tokens:
                v = matrix.get(ti, {}).get(tj, "")
                row.append(f"<td style='text-align:center'>{fmt_num(v)}</td>")
            row.append("</tr>")
            html.append("".join(row))
        # top confusables
        pairs = inter.get("top_confusable_pairs", [])
        if pairs:
            html.append("<p><b>Top confusable pairs:</b> " + ", ".join(
                f"{p.get('a')}-{p.get('b')}:{fmt_num(p.get('similarity',''))}" for p in pairs
            ) + "</p>")

    # Intra clusters
    for ctoken, block in (audit.get("intra", {}) or {}).items():
        html.append(f"<div class='cluster'><h3>Cluster {ctoken} <span class='badge'>n={block.get('count','?')}</span></h3>")
        proto_name = block.get("prototype_name","")
        proto_img = safe_img_b64(thumbs_dir / proto_name) if proto_name else ""
        if proto_img:
            html.append("<div class='row'><div class='card proto'><div class='meta'>PROTOTYPE</div>")
            html.append(f"<img src='{proto_img}' alt='{proto_name}'/>")
            html.append(f"<div class='meta'>{proto_name}</div></div>")
        html.append("<div class='row'>")
        for v in block.get("verdicts", []):
            sname = v.get("sample_name","")
            sim = v.get("similarity","")
            same = v.get("same_symbol", False)
            cd = v.get("color_delta",""); sd = v.get("shape_delta","")
            notes = v.get("notes","")
            b64 = safe_img_b64(thumbs_dir / sname) if sname else ""
            html.append("<div class='card'>")
            if b64:
                html.append(f"<img src='{b64}' alt='{sname}'/>")
            html.append(f"<div class='meta'>{sname}</div>")
            html.append(f"<div class='meta'>same: <b>{bool(same)}</b>, sim: {fmt_num(sim)}</div>")
            html.append(f"<div class='meta'>Δcolor:{cd} Δshape:{sd}</div>")
            if notes:
                html.append(f"<div class='meta'>{notes}</div>")
            html.append("</div>")
        html.append("</div>")  # row
        if block.get("summary"):
            html.append(f"<p><i>{block['summary']}</i></p>")
        html.append("</div>")  # cluster

    html.append("</body></html>")
    with out_html.open("w", encoding="utf-8") as f:
        f.write("\n".join(html))

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Use Gemini to visually audit symbol clusters (intra+inter).")
    ap.add_argument("--codebook", required=True, help="Path to symbol_codebook.json")
    ap.add_argument("--sequences", required=True, help="Path to folder with *.sequence.json")
    ap.add_argument("--thumbs", required=True, help="Path to thumbs_obj (object crops)")
    ap.add_argument("--out", required=True, help="Output folder for audit_report.json/html")
    ap.add_argument("--samples-per-cluster", type=int, default=12, help="Random samples per cluster for intra-audit")
    ap.add_argument("--top-clusters", type=int, default=8, help="Max number of clusters (by size) to audit")
    ap.add_argument("--max-prototypes", type=int, default=12, help="Max prototypes to include in inter-audit")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set (in env or .env)", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    codebook = load_json(Path(args.codebook))
    clusters = build_clusters(codebook)
    clusters = clusters[: args.top_clusters]

    members = collect_members(Path(args.sequences))
    thumbs_dir = Path(args.thumbs)
    out_dir = Path(args.out); ensure_dir(out_dir)

    audit = {"intra": {}, "inter": {}}

    # Intra audits
    for cl in clusters:
        all_names = members.get(cl.cluster_id, [])
        # ensure prototype present (prefer to use codebook's prototype)
        proto = cl.prototype_thumb or (all_names[0] if all_names else "")
        # select samples that are NOT the prototype itself
        candidates = [n for n in all_names if n != proto]
        if len(candidates) == 0:
            continue
        random.seed(42)
        k = min(args.samples_per_cluster, len(candidates))
        samples = random.sample(candidates, k)

        try:
            result = audit_intra_for_cluster(client, cl, samples, thumbs_dir)
        except Exception as e:
            result = {
                "prototype_name": proto,
                "verdicts": [],
                "summary": f"ERROR: {e}"
            }
        result["count"] = len(all_names)
        audit["intra"][cl.token] = result
        print(f"[intra] {cl.token}: {len(samples)} samples audited.")

    # Inter audit on prototypes
    try:
        inter = audit_inter_prototypes(client, clusters, thumbs_dir, max_protos=args.max_prototypes)
    except Exception as e:
        inter = {"tokens": [], "matrix": {}, "top_confusable_pairs": [], "error": str(e)}
    audit["inter"] = inter
    print("[inter] prototype similarity matrix computed.")

    # Save JSON & HTML
    audit_json = out_dir / "audit_report.json"
    with audit_json.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {audit_json}")

    audit_html = out_dir / "audit_report.html"
    render_html(audit, audit_html, thumbs_dir)
    print(f"[OK] wrote {audit_html} (open in your browser)")

if __name__ == "__main__":
    main()
