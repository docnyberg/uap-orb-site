"""Gemini-powered feature extractor for atlas thumbnails.

This utility enriches an existing ``atlas.csv`` file by running every
thumbnail through the Gemini multimodal API and storing the structured
semantic labels returned by the model.  The enriched CSV can then be fed
into :mod:`build_sequences` to drive stricter clustering gates.

Example
-------
    python tools/gemini_feature_extractor.py \
        --atlas public/atlas.csv \
        --thumbs public/thumbs_obj \
        --out public/atlas_gemini.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - dependency error is fatal for runtime use
    raise SystemExit(
        "The google-genai package is required to call the Gemini API. "
        "Install it with `pip install google-genai`."
    ) from exc

try:  # pragma: no cover - UI helper is optional and platform-dependent
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover - tkinter not always available (e.g., headless)
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]

MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

def _configure_client() -> genai.Client:
    """Configure the Gemini client using the GOOGLE_API_KEY."""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        # Fall back to Google Colab secrets when available.
        try:  # pragma: no cover - Colab-specific pathway
            from google.colab import userdata  # type: ignore
        except Exception:  # pragma: no cover - silently ignore outside Colab
            userdata = None
        else:  # pragma: no cover - executed only in Colab
            api_key = userdata.get("GOOGLE_API_KEY")  # type: ignore[attr-defined]

    if not api_key:
        raise SystemExit(
            "GOOGLE_API_KEY is not configured. Set it as an environment variable "
            "or store it in Google Colab userdata before running this script."
        )

    return genai.Client(api_key=api_key)


CLIENT = _configure_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_to_blob(image_path: Path) -> types.Part:
    """Return an image as a google.genai typed Part/Blob."""
    return types.Part(
        inline_data=types.Blob(
            data=image_path.read_bytes(),
            mime_type="image/jpeg",
        )
    )


def _create_dialog_root() -> "tk.Tk":  # type: ignore[name-defined]
    """Return a hidden Tk root window for file dialogs."""
    if tk is None or filedialog is None:  # type: ignore[truthy-function]
        raise RuntimeError(
            "tkinter is unavailable. Provide the required path via command-line "
            "arguments instead of using the picker."
        )

    try:
        root = tk.Tk()  # type: ignore[call-arg]
        root.withdraw()
        return root
    except Exception as exc:  # pragma: no cover - GUI init failures are environment-specific
        raise RuntimeError(
            "Unable to open a file-selection dialog. Provide the path manually using "
            "command-line arguments."
        ) from exc


def _pick_atlas_csv() -> Optional[Path]:
    root = _create_dialog_root()
    try:
        filename = filedialog.askopenfilename(  # type: ignore[union-attr]
            title="Select atlas CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return Path(filename).expanduser() if filename else None


def _pick_thumbs_dir() -> Optional[Path]:
    root = _create_dialog_root()
    try:
        dirname = filedialog.askdirectory(  # type: ignore[union-attr]
            title="Select thumbnail directory",
        )
    finally:
        root.destroy()
    return Path(dirname).expanduser() if dirname else None


def _pick_output_csv() -> Optional[Path]:
    root = _create_dialog_root()
    try:
        filename = filedialog.asksaveasfilename(  # type: ignore[union-attr]
            title="Choose destination for enriched atlas CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return Path(filename).expanduser() if filename else None


def _ensure_path(
    provided: Optional[Path],
    *,
    picker: Callable[[], Optional[Path]],
    descriptor: str,
) -> Path:
    """Return a resolved path, using the picker when ``provided`` is missing."""

    if provided is not None:
        return provided.expanduser()

    try:
        selection = picker()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not selection:
        raise SystemExit(f"No {descriptor} selected; aborting.")

    return selection.expanduser()


def _extract_json_payload(response: Any) -> Dict[str, Any]:
    """Extract a JSON dictionary from a Gemini response object."""
    text = getattr(response, "text", None)
    if not text:
        return {}

    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return {}

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def get_gemini_features(image_blob: types.Part, retries: int = 3) -> Dict[str, Any]:
    """Call Gemini with retry logic and return the structured labels."""
    prompt = (
        "Analyze the object in this image. Describe it using the following JSON schema:\n"
        "{\n"
        "  \"primary_color\": \"The dominant color name (e.g., 'sky_blue', 'lime_green', 'reddish_orange')\",\n"
        "  \"shape_category\": \"One of: 'orb', 'streak', 'fragment', 'nebula', 'ring', 'other'\",\n"
        "  \"texture\": \"A brief description (e.g., 'smooth_glow', 'mottled', 'motion_blur', 'crystalline', 'diffuse')\"\n"
        "}\n"
        "Provide only the raw JSON object as your response."
    )

    for attempt in range(1, retries + 1):
        try:
            response = CLIENT.models.generate_content(
                model=MODEL_NAME,
                contents=types.Content(
                    parts=[types.Part(text=prompt), image_blob]
                ),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            payload = _extract_json_payload(response)
            if payload:
                return payload
        except Exception as exc:  # pragma: no cover - runtime robustness
            print(f"  [WARN] Gemini API call failed on attempt {attempt}: {exc}")
        time.sleep(2)

    return {}


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def enrich_atlas(atlas_path: Path, thumbs_dir: Path, out_path: Path) -> None:
    """Enrich ``atlas_path`` with Gemini features and store it at ``out_path``."""
    df = pd.read_csv(atlas_path)

    if df.empty:
        raise SystemExit(f"Atlas file {atlas_path} has no rows.")

    if "thumb_obj" not in df.columns:
        raise SystemExit("Expected 'thumb_obj' column in atlas CSV.")

    features: List[Dict[str, Any]] = []

    total = len(df)
    thumbs_dir = thumbs_dir.resolve()

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        thumb_name = getattr(row, "thumb_obj", None)
        thumb_path = thumbs_dir / str(thumb_name)
        print(f"Processing {idx}/{total}: {thumb_path.name}…")

        if not thumb_path.exists():
            print("  [WARN] Thumbnail not found; skipping Gemini call.")
            features.append({})
            continue

        try:
            blob = image_to_blob(thumb_path)
        except OSError as exc:
            print(f"  [WARN] Unable to read thumbnail: {exc}")
            features.append({})
            continue

        payload = get_gemini_features(blob)
        features.append(payload)

    enriched = pd.concat([df.reset_index(drop=True), pd.DataFrame(features)], axis=1)

    for col in ("primary_color", "shape_category", "texture"):
        if col in enriched.columns:
            enriched[col] = (
                enriched[col]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .replace({"": "unknown", "nan": "unknown"})
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"\n[OK] Enriched data saved to {out_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich an atlas CSV with Gemini-generated semantic labels.",
    )
    parser.add_argument("--atlas", type=Path, help="Path to the input atlas CSV.")
    parser.add_argument(
        "--thumbs",
        type=Path,
        help="Directory containing the thumbnail images (thumbs_obj).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Destination for the enriched CSV (e.g., atlas_gemini.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    atlas = _ensure_path(
        args.atlas,
        picker=_pick_atlas_csv,
        descriptor="atlas CSV (--atlas)",
    )
    thumbs = _ensure_path(
        args.thumbs,
        picker=_pick_thumbs_dir,
        descriptor="thumbnail directory (--thumbs)",
    )
    out_csv = _ensure_path(
        args.out,
        picker=_pick_output_csv,
        descriptor="output CSV (--out)",
    )

    enrich_atlas(atlas, thumbs, out_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
