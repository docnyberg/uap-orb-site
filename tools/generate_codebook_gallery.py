"""Generate an HTML gallery for symbol codebook prototypes.

Reads ``public/sequences/symbol_codebook_canon.json`` (or any compatible
codebook file) and produces an HTML document that pairs each cluster with its
``thumb.obj`` thumbnail.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from string import Template
from typing import Iterable, List, Mapping


DEFAULT_CANONICAL_CODEBOOK = Path("public/sequences/symbol_codebook_canon.json")
DEFAULT_MERGED_CODEBOOK = Path("public/sequences_merge/symbol_codebook.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codebook",
        type=Path,
        default=DEFAULT_CANONICAL_CODEBOOK,
        help=(
            "Path to the symbol codebook JSON file. Defaults to the canonical "
            "codebook produced by tools/canonicalize_prototypes.py."
        ),
    )
    parser.add_argument(
        "--thumbs-dir",
        type=Path,
        default=Path("public/thumbs_obj"),
        help="Directory that contains the *_obj.jpg thumbnails.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/codebook_gallery.html"),
        help="Destination path for the generated HTML file.",
    )
    return parser.parse_args()


def load_codebook(codebook_path: Path) -> Mapping[str, object]:
    search_paths = [codebook_path]
    if codebook_path == DEFAULT_CANONICAL_CODEBOOK:
        search_paths.append(DEFAULT_MERGED_CODEBOOK)

    for candidate in search_paths:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as fh:
                return json.load(fh)

    search_str = ", ".join(str(p) for p in search_paths)
    raise FileNotFoundError(
        f"Codebook file not found. Looked for: {search_str}. "
        "Pass --codebook to point at the correct file."
    )


def normalise_clusters(raw_clusters: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    clusters: List[Mapping[str, object]] = []
    for cluster in raw_clusters:
        if not isinstance(cluster, Mapping):
            continue
        prototype = cluster.get("prototype", {})
        if not isinstance(prototype, Mapping):
            prototype = {}
        clusters.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "token": cluster.get("token"),
                "count": cluster.get("count"),
                "thumb_obj": prototype.get("thumb_obj"),
            }
        )
    clusters.sort(key=lambda c: (c.get("cluster_id"), str(c.get("token"))))
    return clusters


def extract_clusters(codebook: Mapping[str, object]) -> List[Mapping[str, object]]:
    clusters = codebook.get("clusters")
    if isinstance(clusters, list):
        return normalise_clusters(clusters)

    legend = codebook.get("legend")
    if isinstance(legend, Mapping):
        converted = []
        for token, entry in legend.items():
            if not isinstance(entry, Mapping):
                continue
            cluster_payload = dict(entry)
            cluster_payload.setdefault("token", token)
            converted.append(cluster_payload)
        return normalise_clusters(converted)

    raise ValueError(
        "Unexpected codebook format: expected a 'clusters' list or 'legend' mapping."
    )


def build_html(clusters: Iterable[Mapping[str, object]], thumbs_dir: Path) -> str:
    thumbs_dir = thumbs_dir.resolve()
    thumbs_rel = thumbs_dir.name
    rows: List[str] = []
    for cluster in clusters:
        thumb_name = cluster.get("thumb_obj")
        thumb_path = f"{thumbs_rel}/{thumb_name}" if thumb_name else None
        thumb_missing = bool(thumb_name) and not (thumbs_dir / thumb_name).exists()

        rows.append(
            "    <article class=\"cluster\">\n"
            f"      <header><h2>Cluster {html.escape(str(cluster.get('cluster_id')))}</h2></header>\n"
            "      <div class=\"media\">\n"
            + (
                f"        <img src=\"{html.escape(thumb_path)}\" alt=\"Cluster {html.escape(str(cluster.get('cluster_id')))} thumbnail\">\n"
                if thumb_path and not thumb_missing
                else "        <div class=\"missing\">Missing thumb.obj</div>\n"
            )
            + "      </div>\n"
            "      <dl class=\"meta\">\n"
            f"        <dt>Token</dt><dd>{html.escape(str(cluster.get('token')))}</dd>\n"
            f"        <dt>Instances</dt><dd>{html.escape(str(cluster.get('count')))}</dd>\n"
            + (
                f"        <dt>Thumb file</dt><dd>{html.escape(thumb_name)}</dd>\n"
                if thumb_name
                else ""
            )
            + (
                "        <dt>Status</dt><dd class=\"warning\">Missing thumbnail file</dd>\n"
                if thumb_missing
                else ""
            )
            + "      </dl>\n"
            "    </article>"
        )

    template = Template(
        """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Symbol Codebook Gallery</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #111;
        color: #f2f2f2;
      }
      body {
        margin: 0;
        padding: 2rem;
        background: radial-gradient(circle at top, #1d2733, #05070b 60%);
      }
      h1 {
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
      }
      .cluster {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        background: rgba(13, 17, 23, 0.85);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 120ms ease, box-shadow 200ms ease;
      }
      .cluster:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.4);
      }
      .cluster header {
        margin-bottom: 0.75rem;
      }
      .cluster h2 {
        margin: 0;
        font-size: 1.1rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      .media {
        width: 100%;
        aspect-ratio: 1 / 1;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 0.75rem;
        background: rgba(255, 255, 255, 0.04);
      }
      .media img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
      }
      .media .missing {
        font-size: 0.85rem;
        color: rgba(255, 230, 150, 0.9);
        padding: 0.5rem;
        text-align: center;
      }
      .meta {
        margin: 0;
        display: grid;
        grid-template-columns: auto 1fr;
        column-gap: 0.75rem;
        row-gap: 0.4rem;
        font-size: 0.9rem;
      }
      .meta dt {
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
      }
      .meta dd {
        margin: 0;
      }
      .meta .warning {
        color: #ffb65c;
      }
      footer {
        margin-top: 2rem;
        text-align: center;
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.55);
      }
    </style>
  </head>
  <body>
    <h1>Symbol Codebook Gallery</h1>
    <section class="grid">
$rows
    </section>
    <footer>
      Generated by tools/generate_codebook_gallery.py
    </footer>
  </body>
</html>
"""
    )
    return template.substitute(rows="\n".join(rows))


def main() -> None:
    args = parse_args()
    codebook = load_codebook(args.codebook)

    normalised = extract_clusters(codebook)
    html_content = build_html(normalised, args.thumbs_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_content, encoding="utf-8")
    print(f"[ok] Wrote gallery → {args.output}")


if __name__ == "__main__":
    main()

