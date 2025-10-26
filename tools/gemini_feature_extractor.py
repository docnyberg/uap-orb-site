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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from contextlib import suppress

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


def _extract_text(resp) -> str:
    """Return best-effort text from a google-genai response. Handles empty .text and parts."""
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    out: List[str] = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if content and getattr(content, "parts", None):
            for p in content.parts:
                pt = getattr(p, "text", None)
                if isinstance(pt, str) and pt:
                    out.append(pt)
    return "".join(out).strip()


def _extract_json_payload_from_text(text: str) -> Dict[str, Any]:
    """Extract a JSON dictionary from Gemini response text."""
    if not text:
        return {}

    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return {}

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _call_with_timeout(fn, timeout_s=60):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout:
            return {"__timeout__": True}


def _safe_gemini_generate(prompt_text, image_blob, timeout_s=60, max_retries=2, backoff=2.0):
    """
    Call Gemini with timeout and simple retry/backoff. Returns either a response object
    or {"status":"timeout"} after exhausting retries.
    """
    attempt = 0
    while True:
        def _go():
            return CLIENT.models.generate_content(
                model=MODEL_NAME,
                contents=types.Content(parts=[types.Part(text=prompt_text), image_blob]),
                config=types.GenerateContentConfig(temperature=0.0),
            )

        res = _call_with_timeout(_go, timeout_s=timeout_s)
        if isinstance(res, dict) and res.get("__timeout__"):
            attempt += 1
            if attempt > max_retries:
                return {"status": "timeout"}
            time.sleep(backoff ** attempt)
            continue
        return res


try:
    _ckpt_write
except NameError:
    def _ckpt_write(path, index, key, status, payload):
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "index": index,
                    "key": key,
                    "status": status,
                    "payload": payload,
                })
                + "\n"
            )


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

PROMPT_TEXT = (
    "Analyze the object in this image. Describe it using the following JSON schema:\n"
    "{\n"
    "  \"primary_color\": \"The dominant color name (e.g., 'sky_blue', 'lime_green', 'reddish_orange')\",\n"
    "  \"shape_category\": \"One of: 'orb', 'streak', 'fragment', 'nebula', 'ring', 'other'\",\n"
    "  \"texture\": \"A brief description (e.g., 'smooth_glow', 'mottled', 'motion_blur', 'crystalline', 'diffuse')\"\n"
    "}\n"
    "Provide only the raw JSON object as your response."
)


def enrich_atlas(
    atlas_path: Path,
    thumbs_dir: Path,
    out_path: Path,
    *,
    timeout_seconds: int = 60,
    max_retries: int = 2,
    retry_timeouts: bool = False,
) -> None:
    """Enrich ``atlas_path`` with Gemini features and store it at ``out_path``."""
    df = pd.read_csv(atlas_path)

    if df.empty:
        raise SystemExit(f"Atlas file {atlas_path} has no rows.")

    if "thumb_obj" not in df.columns:
        raise SystemExit("Expected 'thumb_obj' column in atlas CSV.")

    features: List[Dict[str, Any]] = []

    total = len(df)
    thumbs_dir = thumbs_dir.resolve()

    ckpt_path = out_path.with_name(out_path.name + ".ckpt.jsonl")
    processed: Dict[int, Dict[str, Any]] = {}
    if ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed[int(rec["index"])] = rec
                except Exception:
                    pass

    try:
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            thumb_name = getattr(row, "thumb_obj", None)
            thumb_path = thumbs_dir / str(thumb_name)
            print(f"Processing {idx}/{total}: {thumb_path.name}…")

            idx0 = idx - 1
            key = str(thumb_name)
            prev = processed.get(idx0)
            if prev and prev.get("key") != key:
                prev = None

            if prev:
                allowed_statuses = ["ok"] if retry_timeouts else ["ok", "timeout"]
                if prev.get("status") in allowed_statuses:
                    features.append(prev.get("payload", {}))
                    continue

            if not thumb_path.exists():
                scene_dir = Path(str(thumbs_dir).replace("thumbs_obj", "thumbs"))
                cand = scene_dir / thumb_path.name.replace("_obj", "")
                if scene_dir.exists() and cand.exists():
                    thumb_path = cand
                else:
                    print("  [WARN] Thumbnail not found; skipping.")
                    features.append({})
                    _ckpt_write(ckpt_path, idx0, key, "missing", {})
                    continue

            try:
                blob = image_to_blob(thumb_path)
            except OSError as exc:
                print(f"  [WARN] Unable to read thumbnail: {exc}")
                features.append({})
                _ckpt_write(ckpt_path, idx0, key, "ioerror", {})
                continue

            resp = _safe_gemini_generate(
                prompt_text=PROMPT_TEXT,
                image_blob=blob,
                timeout_s=timeout_seconds,
                max_retries=max_retries,
            )
            if isinstance(resp, dict) and resp.get("status") == "timeout":
                print("  [WARN] Gemini timeout; skipping.")
                features.append({})
                _ckpt_write(ckpt_path, idx0, key, "timeout", {})
                continue

            text = _extract_text(resp)
            if not text:
                print("  [WARN] Empty response; continuing.")
                features.append({})
                _ckpt_write(ckpt_path, idx0, key, "empty", {})
                continue

            payload = _extract_json_payload_from_text(text)
            features.append(payload)

            _ckpt_write(ckpt_path, idx0, key, "ok", payload)

            if idx % 200 == 0:
                partial = pd.concat(
                    [df.iloc[:idx].reset_index(drop=True), pd.DataFrame(features)],
                    axis=1,
                )
                partial_out = out_path.with_suffix(".partial.csv")
                partial.to_csv(partial_out, index=False)
                print(f"[checkpoint] wrote {partial_out}")
    except KeyboardInterrupt:
        partial = pd.concat(
            [df.iloc[: len(features)].reset_index(drop=True), pd.DataFrame(features)],
            axis=1,
        )
        partial_out = out_path.with_suffix(".partial.csv")
        partial.to_csv(partial_out, index=False)
        print(f"\n[INTERRUPTED] Wrote partial to {partial_out}. You can resume later.")
        sys.exit(130)

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
    with suppress(Exception):
        ckpt_path.unlink()
        out_path.with_suffix(".partial.csv").unlink()
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
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-item Gemini call timeout (seconds)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per item when Gemini times out",
    )
    parser.add_argument(
        "--retry-timeouts",
        action="store_true",
        help="Retry rows previously marked as status=timeout",
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

    enrich_atlas(
        atlas,
        thumbs,
        out_csv,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_timeouts=args.retry_timeouts,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
