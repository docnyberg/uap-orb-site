#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sequences.py — strict‑gated clustering + sequence builder

Usage:
  python build_sequences.py --atlas "C:\\path\\to\\atlas.csv" \
    --thumbs "C:\\path\\to\\thumbs_obj" \
    --out "C:\\path\\to\\sequences" \
    [--min-samples 6] [--eps 0.28] \
    [--w-phash 2.0] [--w-hue 1.0] [--w-feat 1.0] \
    [--gate-hue-deg 14] [--gate-area-ratio 0.45] \
    [--gate-solidity 0.20] [--gate-ecc 0.28] [--gate-ar 0.60] \
    [--min-area 50]

Notes
- Expects atlas.csv columns produced by your pipeline (e.g., thumb_obj, json_file, e_idx/eid, hsv_mean_h/s/v, area, solidity, eccentricity, aspect_ratio, phash64).
- If some features missing, will compute best-effort from thumbs.
- Strict gates guarantee clusters won’t mix clearly different shapes/colors.
"""

import argparse
import json
import math
import os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# Optional; used only if we need to compute missing features
try:
    import cv2
except Exception:
    cv2 = None

from sklearn.cluster import DBSCAN

# ---------------------------
# Helpers
# ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        if isinstance(x, str) and not x.strip():
            return default
        return float(x)
    except Exception:
        return default

def parse_phash64(x):
    """
    Accepts hex string for 64-bit pHash (e.g., 'e3a1...'), int, or None.
    Returns Python int or None.
    """
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(s, 16)
    except Exception:
        # Sometimes csv dumps as decimal; last attempt
        try:
            return int(s)
        except Exception:
            return None

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def circ_hue_delta(h1, h2, scale=180.0):
    """Circular hue diff with wraparound (OpenCV HSV: 0..180)."""
    if np.isnan(h1) or np.isnan(h2):
        return np.nan
    d = abs(h1 - h2)
    if d > scale/2:
        d = scale - d
    return d

def robust_z(x):
    """Return robust z-score using median & MAD (scaled)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

def quantize_hue(h, bin_deg=6.0):
    if np.isnan(h):
        return None
    return int(round(h / bin_deg))

def medoid_index(dist_matrix):
    """Return index of medoid (min sum of distances)."""
    if dist_matrix.size == 0:
        return None
    sums = dist_matrix.sum(axis=1)
    return int(np.argmin(sums))

def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

        # ---------------------------
# Color / morphology helpers
# ---------------------------

# Tunables (you can later promote to CLI if you want)
SAT_MIN        = 0.18   # if S < SAT_MIN → "Unknown"
V_DARK_MIN     = 0.05   # very dark = effectively unknown/neutral
EDGE_FRAC_FLASH= 0.12   # edge density fraction that suggests a burst
V_FLASH_MIN    = 0.60   # minimum V to treat an edge spike as flash
RING_SCORE_MIN = 0.25   # annulus brighter than center by ≥25% → ring

# 4 bins that cover the full 360° hue circle, with wrap-around
# Yellow: 30–90°, Green: 90–160°, Blue: 160–260°, Pink/Magenta: 260–360° + 0–30°
COLOR_BINS = [
    ("Yellow",       30.0,  90.0),
    ("Green",        90.0, 160.0),
    ("Blue",        160.0, 260.0),
    ("Pink/Magenta", 260.0, 360.0),
    ("Pink/Magenta",   0.0,  30.0),  # wrap segment
]

def label_color_hsv(h, s, v, sat_min=SAT_MIN, v_dark=V_DARK_MIN):
    """Map HSV → one of 4 full-name color bins; reserve 'Unknown' for low S or very low V."""
    if np.isnan(h) or np.isnan(s) or np.isnan(v):
        return "Unknown"
    if s < sat_min or v < v_dark:
        return "Unknown"
    h = float(h) % 360.0
    for name, lo, hi in COLOR_BINS:
        if lo <= h < hi:
            return name
    return "Unknown"

def ring_score_center_vs_annulus(gray: np.ndarray) -> float:
    """
    Simple ring detector: compare mean intensity in a small center disk vs an annulus.
    Returns a score in [0, 1]-ish: max(annulus - center, 0) / max(annulus, 1).
    """
    hh, ww = gray.shape[:2]
    cy, cx = hh // 2, ww // 2
    R  = int(min(hh, ww) * 0.40)     # outer radius for our analysis window
    r1 = max(1, int(R * 0.25))       # center radius
    r2 = max(r1 + 1, int(R * 0.50))  # annulus outer radius

    yy, xx = np.ogrid[:hh, :ww]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    center_mask  = (dist <= r1)
    annulus_mask = (dist >  r1) & (dist <= r2)

    center_mean  = float(np.mean(gray[center_mask])) if np.any(center_mask) else 0.0
    annulus_mean = float(np.mean(gray[annulus_mask])) if np.any(annulus_mask) else 0.0
    if annulus_mean <= 1.0:  # avoid div-by-zero and tiny denominators
        return 0.0
    return max(annulus_mean - center_mean, 0.0) / annulus_mean

# ---------------------------
# Feature extraction (fallbacks if missing)
# ---------------------------

def compute_basic_shape_from_img(img_bgr):
    """
    Compute area, aspect_ratio, solidity, eccentricity (best-effort)
    from a single-channel mask extracted via adaptive threshold.
    Returns dict of features; NaNs if cv2 not available.
    """
    if cv2 is None or img_bgr is None:
        return dict(area=np.nan, aspect_ratio=np.nan, solidity=np.nan, eccentricity=np.nan)
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # adaptive threshold to get a blob
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 35, 2)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return dict(area=np.nan, aspect_ratio=np.nan, solidity=np.nan, eccentricity=np.nan)
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            return dict(area=np.nan, aspect_ratio=np.nan, solidity=np.nan, eccentricity=np.nan)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h) if h > 0 else np.nan

        # solidity = area/convex_area
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else np.nan
        solidity = area / hull_area if hull_area and hull_area > 0 else np.nan

        # eccentricity via fitEllipse if possible
        ecc = np.nan
        if len(cnt) >= 5:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)  # MA=major axis, ma=minor axis
            MA = float(MA); ma = float(ma)
            if MA > 0 and ma > 0:
                # ellipse eccentricity e = sqrt(1 - (b^2/a^2)); a = max(MA, ma)/2, b = min/2
                a = max(MA, ma)/2.0
                b = min(MA, ma)/2.0
                if a > 0:
                    ratio = (b*b)/(a*a)
                    ratio = max(0.0, min(1.0, ratio))  # clamp for numeric safety
                    ecc = math.sqrt(1.0 - ratio)
        return dict(area=area, aspect_ratio=aspect_ratio, solidity=solidity, eccentricity=ecc)
    except Exception:
        return dict(area=np.nan, aspect_ratio=np.nan, solidity=np.nan, eccentricity=np.nan)

def compute_phash64_from_img(img_bgr):
    """
    Compute a 64-bit pHash (integer) using DCT (no external deps).
    """
    if cv2 is None or img_bgr is None:
        return None
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        gray = np.float32(gray)
        dct = cv2.dct(gray)
        # top-left 8x8
        dct8 = dct[:8, :8]
        med = np.median(dct8)
        bits = (dct8 > med).astype(np.uint8).flatten()
        # pack into 64-bit
        v = 0
        for b in bits:
            v = (v << 1) | int(b)
        return v
    except Exception:
        return None

def load_img(thumbs_dir: Path, name: str):
    if cv2 is None:
        return None
    try:
        p = (thumbs_dir / name)
        if not p.exists():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# ---------------------------
# Strict-gated distance
# ---------------------------

class StrictGatedMetric:
    def __init__(self, w_phash=2.0, w_hue=1.0, w_feat=1.0,
                 hue_gate_deg=14.0, area_ratio_gate=0.45,
                 solidity_gate=0.20, ecc_gate=0.28, ar_gate=0.60):
        self.w_phash = float(w_phash)
        self.w_hue = float(w_hue)
        self.w_feat = float(w_feat)
        self.hue_gate_deg = float(hue_gate_deg)
        self.area_ratio_gate = float(area_ratio_gate)
        self.solidity_gate = float(solidity_gate)
        self.ecc_gate = float(ecc_gate)
        self.ar_gate = float(ar_gate)

    def pair_dist(self, a, b, feat_scale):
        """
        a, b are dict-like records with required keys:
        'phash64' (int or None), 'h' (hue), 'area','solidity','ecc','ar'
        feat_scale: dict with 'log_area_mu','log_area_sigma', etc for z-scaling.
        Returns distance in [0,1].
        """
        # Hard gates
        # Hue
        dh = circ_hue_delta(a['h'], b['h'], scale=180.0)
        if not np.isnan(dh) and dh > self.hue_gate_deg:
            return 1.0

        # Area ratio
        A1, A2 = a['area'], b['area']
        if (not np.isnan(A1)) and (not np.isnan(A2)):
            mn = min(A1, A2); mx = max(A1, A2) if max(A1, A2) > 0 else np.nan
            if not np.isnan(mx):
                if (mn / mx) < self.area_ratio_gate:
                    return 1.0

        # Solidity
        s1, s2 = a['solidity'], b['solidity']
        if (not np.isnan(s1)) and (not np.isnan(s2)):
            if abs(s1 - s2) > self.solidity_gate:
                return 1.0

        # Eccentricity
        e1, e2 = a['ecc'], b['ecc']
        if (not np.isnan(e1)) and (not np.isnan(e2)):
            if abs(e1 - e2) > self.ecc_gate:
                return 1.0

        # Aspect ratio
        r1, r2 = a['ar'], b['ar']
        if (not np.isnan(r1)) and (not np.isnan(r2)):
            if abs(r1 - r2) > self.ar_gate:
                return 1.0

        # Soft distances
        parts = []
        weights = []

        # pHash
        d_ph = 1.0
        if a['phash64'] is not None and b['phash64'] is not None:
            d_ph = hamming64(a['phash64'], b['phash64']) / 64.0
        parts.append(d_ph); weights.append(self.w_phash)

        # Hue
        d_hue = 0.0 if np.isnan(dh) else (dh / 180.0)
        parts.append(d_hue); weights.append(self.w_hue)

        # Shape features L1 (z-scored)
        f_sum = 0.0; f_cnt = 0
        # log_area
        for key, mu_key, sd_key in [
            ('log_area', 'log_area_mu', 'log_area_sd'),
            ('solidity', 'sol_mu', 'sol_sd'),
            ('ecc', 'ecc_mu', 'ecc_sd'),
            ('ar', 'ar_mu', 'ar_sd'),
        ]:
            va = a[key]; vb = b[key]
            mu = feat_scale[mu_key]; sd = feat_scale[sd_key]
            if not np.isnan(va) and not np.isnan(vb):
                za = 0.0 if sd == 0 or np.isnan(sd) else (va - mu)/sd
                zb = 0.0 if sd == 0 or np.isnan(sd) else (vb - mu)/sd
                f_sum += abs(za - zb)
                f_cnt += 1
        d_feat = (f_sum / f_cnt) if f_cnt > 0 else 0.0
        # map to 0..1 via logistic-ish squashing (keeps moderate values in)
        d_feat = 1.0 - math.exp(-d_feat)  # monotonic, 0→0, grows to 1
        parts.append(d_feat); weights.append(self.w_feat)

        denom = sum(weights) if sum(weights) > 0 else 1.0
        d = sum(p*w for p, w in zip(parts, weights)) / denom
        return float(max(0.0, min(1.0, d)))

# ---------------------------
# Load atlas & prepare features
# ---------------------------

def load_atlas(atlas_csv: Path, thumbs_dir: Path, min_area_px: int):
    """
    Robust atlas reader:
      - Accepts many alternate header names for: thumbnail, clip/json id, event index, HSV, shape, phash.
      - If event index is missing, derives it from the thumbnail filename (e###/frame###/obj_frame###),
        else by per-clip rank of start_ts, else by per-clip row order.
      - Back-fills missing HSV/shape/pHash directly from the image when available.
      - Applies an optional minimum-area filter.
      - Computes log_area for downstream normalization.
    Returns: list[dict] with keys:
      thumb, json_file, e_idx, h, s, v, area, solidity, ecc, ar, phash64, log_area
    """
    import os, re
    import pandas as pd
    import numpy as np

    df = pd.read_csv(atlas_csv)
    if df.empty:
        raise ValueError("atlas.csv is empty")

    # Map lower->original for flexible lookup
    colmap = {c.strip().lower(): c for c in df.columns}

    def pick(*cands) -> str | None:
        for c in cands:
            if c and c.lower() in colmap:
                return colmap[c.lower()]
        return None

    # Likely names in atlas
    col_thumb = pick("thumb_obj", "thumb", "thumb_path", "obj_thumb", "obj", "thumbname", "thumb_name")
    col_json  = pick("json_file", "json", "video_file", "video", "source_json", "file", "src", "clip")
    col_event = pick("event_index", "event_idx", "e_idx", "eid", "e_id", "eindex", "index", "idx",
                     "frame_idx", "frame_index", "frame", "slice_index")

    col_start = pick("start_ts", "start", "t0", "begin_ts", "ts_start")
    col_end   = pick("end_ts", "end", "t1", "stop_ts", "ts_end")

    col_h = pick("h_deg_img", "h_deg", "hsv_mean_h", "hue_deg", "h")   # degrees (0..360)
    col_s = pick("s_norm", "hsv_mean_s", "s")                           # 0..1
    col_v = pick("v_norm", "hsv_mean_v", "v")                           # 0..1

    col_area = pick("area", "area_px")
    col_sol  = pick("solidity")
    col_ecc  = pick("eccentricity", "ecc")
    col_ar   = pick("aspect_ratio", "ar")
    col_phash = pick("phash64", "phash")

    if not col_thumb or not col_json:
        raise ValueError(
            "atlas.csv must include a thumbnail column and a clip id column.\n"
            f"Looked for thumb_obj/thumb/... and json_file/video/...; got: {list(df.columns)}"
        )

    # Derive event index if missing
    if not col_event:
        def infer_idx_from_thumb(name: str) -> int | None:
            s = os.path.basename(str(name))
            for pat in (r"_e(\d+)_", r"e(\d+)", r"event[_\-]?(\d+)", r"frame[_\-]?(\d+)", r"obj_frame[_\-]?(\d+)"):
                m = re.search(pat, s, flags=re.IGNORECASE)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        pass
            return None

        df["_e_idx_inferred"] = df[col_thumb].apply(infer_idx_from_thumb)

        if df["_e_idx_inferred"].isna().all() and col_start:
            # Rank by start_ts per clip
            try:
                df["_e_idx_inferred"] = (
                    df.groupby(df[col_json])[col_start]
                      .rank(method="first")
                      .astype("Int64")
                      .fillna(0)
                      .astype(int) - 1
                )
            except Exception:
                df["_e_idx_inferred"] = df.groupby(df[col_json]).cumcount()
        elif df["_e_idx_inferred"].isna().all():
            # Stable per-clip row order
            df["_e_idx_inferred"] = df.groupby(df[col_json]).cumcount()

        col_event = "_e_idx_inferred"

    # Build records
    recs = []
    for _, row in df.iterrows():
        thumb = str(row[col_thumb]) if (col_thumb and not pd.isna(row[col_thumb])) else None
        json_file = str(row[col_json]) if (col_json and not pd.isna(row[col_json])) else None
        e_idx = int(row[col_event]) if (col_event and not pd.isna(row[col_event])) else 0

        h = safe_float(row[col_h]) if col_h else np.nan
        s = safe_float(row[col_s]) if col_s else np.nan
        v = safe_float(row[col_v]) if col_v else np.nan

        area = safe_float(row[col_area]) if col_area else np.nan
        sol  = safe_float(row[col_sol])  if col_sol  else np.nan
        ecc  = safe_float(row[col_ecc])  if col_ecc  else np.nan
        ar   = safe_float(row[col_ar])   if col_ar   else np.nan

        ph   = parse_phash64(row[col_phash]) if col_phash else None

        recs.append(dict(
            idx=e_idx,
            thumb=thumb,
            json_file=json_file,
            e_idx=e_idx,
            h=h, s=s, v=v,
            area=area, solidity=sol, ecc=ecc, ar=ar,
            phash64=ph,
            # include timestamps if your atlas has them
            start_ts=safe_float(row[col_start]) if col_start else None,
            end_ts=safe_float(row[col_end]) if col_end else None,
        ))

    # Determine if we need image-derived backfills
    need_img_fill = any(
        (np.isnan(r["h"]) or np.isnan(r["s"]) or np.isnan(r["v"])
         or np.isnan(r["area"]) or np.isnan(r["solidity"]) or np.isnan(r["ecc"]) or np.isnan(r["ar"])
         or r["phash64"] is None)
        for r in recs
    )

    # Back-fill from images when available (HSV, shape, phash)
    if need_img_fill and cv2 is not None:
        for r in recs:
            if not r["thumb"]:
                continue
            img = load_img(thumbs_dir, r["thumb"])
            if img is None:
                continue

            # HSV backfill
            if np.isnan(r["h"]) or np.isnan(r["s"]) or np.isnan(r["v"]):
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                H, S, V = cv2.split(hsv)
                if np.isnan(r["h"]): r["h"] = float(H.mean()) * 2.0
                if np.isnan(r["s"]): r["s"] = float(S.mean()) / 255.0
                if np.isnan(r["v"]): r["v"] = float(V.mean()) / 255.0

            # Shape backfill
            if np.isnan(r["area"]) or np.isnan(r["solidity"]) or np.isnan(r["ecc"]) or np.isnan(r["ar"]):
                feats = compute_basic_shape_from_img(img)
                if np.isnan(r["area"])      and not np.isnan(feats.get("area", np.nan)):           r["area"]     = float(feats["area"])
                if np.isnan(r["ar"])        and not np.isnan(feats.get("aspect_ratio", np.nan)):   r["ar"]       = float(feats["aspect_ratio"])
                if np.isnan(r["solidity"])  and not np.isnan(feats.get("solidity", np.nan)):       r["solidity"] = float(feats["solidity"])
                if np.isnan(r["ecc"])       and not np.isnan(feats.get("eccentricity", np.nan)):   r["ecc"]      = float(feats["eccentricity"])

            # pHash backfill
            if r["phash64"] is None:
                ph = compute_phash64_from_img(img)
                if ph is not None:
                    r["phash64"] = ph

        # ---- Enhanced annotations (always run) ----
    for r in recs:
        # log_area
        r["log_area"] = np.log(max(1.0, r["area"])) if not np.isnan(r["area"]) else np.nan

        # Base morphology via solidity (quick triage)
        if not np.isnan(r["solidity"]):
            if r["solidity"] >= 0.85:
                morph = "sphere"
            elif r["solidity"] >= 0.50:
                morph = "ring"   # candidate ring; will verify with center/annulus test
            else:
                morph = "cross/star"
        else:
            morph = "unknown"

        # Load image once for per-event visual tests
        img = load_img(thumbs_dir, r["thumb"]) if r.get("thumb") else None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if (img is not None and cv2 is not None) else None

        # Optional: refine ring via center vs annulus brightness
        if gray is not None:
            rs = ring_score_center_vs_annulus(gray)
            r["ring_score"] = float(rs)
            if rs >= RING_SCORE_MIN:
                morph = "ring"
        else:
            r["ring_score"] = float("nan")

        # Orientation (yin-yang) for all shapes (↑ top-bright, ↓ bottom-bright)
        ori = None
        if gray is not None:
            hh, ww = gray.shape
            top_mean = float(np.mean(gray[:hh // 2, :]))
            bot_mean = float(np.mean(gray[hh // 2:, :]))
            if top_mean > bot_mean * 1.10:
                ori = "↑"
            elif bot_mean > top_mean * 1.10:
                ori = "↓"

        # Edge density for flash detection
        edge_frac = float("nan")
        if gray is not None:
            edges = cv2.Canny(gray, 50, 150)
            edge_frac = float(np.count_nonzero(edges)) / float(edges.size)

        # Robust flash: very bright OR (edge spike + moderately bright)
        v_val = r.get("v", np.nan)
        flash = (not np.isnan(v_val) and v_val >= 0.90)  # absolute bright spike
        if (not flash) and (not np.isnan(edge_frac)) and (edge_frac >= EDGE_FRAC_FLASH) and (not np.isnan(v_val)) and (v_val >= V_FLASH_MIN):
            flash = True

        # Full-circle color label (Unknown reserved for low S or very low V)
        r["color_label"] = label_color_hsv(r.get("h", np.nan), r.get("s", np.nan), r.get("v", np.nan))

        # Commit fields
        r["morphology"] = morph
        r["orientation"] = ori
        r["edge_frac"] = edge_frac
        r["flash"] = bool(flash)

    return recs


# ---------------------------
# Clustering (strict‑gated)
# ---------------------------

def cluster_records_strict(recs, args):
    """
    Cluster with strict gating. IMPORTANT: -1 (DBSCAN noise) stays noise (cluster=None).
    Very small clusters (< args.min_cluster_size) remain clusters here; we skip them in tokenization.
    """
    import numpy as np
    from collections import defaultdict

    # Prepare global z-scale for shape features
    arr_logA = np.array([r['log_area'] for r in recs], dtype=float)
    arr_sol  = np.array([r['solidity'] for r in recs], dtype=float)
    arr_ecc  = np.array([r['ecc'] for r in recs], dtype=float)
    arr_ar   = np.array([r['ar'] for r in recs], dtype=float)

    feat_scale = dict(
        log_area_mu=np.nanmean(arr_logA), log_area_sd=np.nanstd(arr_logA),
        sol_mu=np.nanmean(arr_sol),       sol_sd=np.nanstd(arr_sol),
        ecc_mu=np.nanmean(arr_ecc),       ecc_sd=np.nanstd(arr_ecc),
        ar_mu=np.nanmean(arr_ar),         ar_sd=np.nanstd(arr_ar),
    )

    metric = StrictGatedMetric(
        w_phash=args.w_phash, w_hue=args.w_hue, w_feat=args.w_feat,
        hue_gate_deg=args.gate_hue_deg, area_ratio_gate=args.gate_area_ratio,
        solidity_gate=args.gate_solidity, ecc_gate=args.gate_ecc, ar_gate=args.gate_ar
    )

    # Hue buckets
    hue_bins = defaultdict(list)
    for r in recs:
        hb = quantize_hue(r['h'], bin_deg=6.0) if not np.isnan(r['h']) else ('nan',)
        hue_bins[hb].append(r)

    next_cluster_id = 0
    for hb, bucket in hue_bins.items():
        n = len(bucket)
        if n == 1:
            # don't assign a cluster yet; treat as tiny/noise-like; tokenization will skip it
            bucket[0]['cluster'] = None
            continue

        # pairwise distances in bucket
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i+1, n):
                D[i, j] = D[j, i] = metric.pair_dist(bucket[i], bucket[j], feat_scale)

        # DBSCAN
        db = DBSCAN(eps=float(args.eps), min_samples=int(args.min_samples), metric='precomputed')
        labels = db.fit_predict(D)

        # Assign clusters; keep noise as None
        unique = np.unique(labels)
        for li in unique:
            idxs = np.where(labels == li)[0]
            if li == -1:
                for k in idxs:
                    bucket[k]['cluster'] = None
                continue
            # select medoid within this component
            subD = D[np.ix_(idxs, idxs)]
            mloc = np.argmin(subD.sum(axis=1))
            med_k = int(idxs[int(mloc)])
            gid = next_cluster_id
            next_cluster_id += 1
            for k in idxs:
                bucket[k]['cluster'] = gid
            # tag a prototype
            for k in idxs:
                bucket[k]['is_prototype'] = (k == med_k)

    # Gather clusters (exclude None)
    by_cluster = defaultdict(list)
    for r in recs:
        cid = r.get('cluster', None)
        if cid is not None:
            by_cluster[cid].append(r)

    # If a cluster has no prototype tag (rare), mark its first as prototype
    for cid, rows in by_cluster.items():
        if not any(row.get('is_prototype') for row in rows):
            rows[0]['is_prototype'] = True

    return recs, by_cluster, feat_scale


# ---------------------------
# Tokenization & codebook
# ---------------------------

def assign_tokens(by_cluster, min_cluster_size: int = 3):
    """
    Assign tokens to clusters, skipping tiny clusters (treated as noise later).
    Uses an unbounded base-26 stream: A..Z, AA..AZ, BA..BZ, ... AAA.., etc.
    Returns (token_map: cid->token, legend: token->meta)
    """
    import string
    def base26_stream():
        letters = string.ascii_uppercase
        n = 1
        while True:
            # emit all length-n strings
            total = len(letters) ** n
            for idx in range(total):
                x = idx
                s = []
                for _ in range(n):
                    s.append(letters[x % 26])
                    x //= 26
                yield "".join(reversed(s))
            n += 1

    # clusters sorted by size (desc), then by prototype hue if available
    clusters = []
    for cid, rows in by_cluster.items():
        if not rows:
            continue
        cnt = len(rows)
        proto = next((r for r in rows if r.get('is_prototype')), rows[0])
        clusters.append((cid, cnt, proto))
    clusters.sort(key=lambda x: (-x[1], (x[2].get('h', 999.0) if x[2].get('h') is not None else 999.0)))

    token_map = {}
    legend = {}
    gen = base26_stream()
    for cid, cnt, proto in clusters:
        if cnt < min_cluster_size:
            # leave un-tokenized; will render as 'X' later
            continue
        tok = next(gen)  # unlimited supply
        token_map[cid] = tok
        legend[tok] = dict(
            cluster_id=int(cid),
            count=int(cnt),
            prototype=dict(
                thumb_obj=proto.get('thumb'),
                json_file=proto.get('json_file'),
                e_idx=proto.get('e_idx'),
                h=proto.get('h'), s=proto.get('s'), v=proto.get('v'),
                area=proto.get('area'), solidity=proto.get('solidity'),
                ecc=proto.get('ecc'), ar=proto.get('ar'),
                phash64=proto.get('phash64')
            )
        )
    return token_map, legend


# ---------------------------
# Sequence builder
# ---------------------------

def build_sequences(recs, token_map):
    """
    For each video json_file: sort by event index and assign the majority token
    among thumbs that share that event index. Also produce:
      - binary_seq (V-based)
      - symbol_seq_enhanced (full color names + ↑/↓ + ✦)
      - events list (per-event details, including cluster_id and thumb_obj)
      - cycles list (split by >2s gap OR timestamp reset; always a list)
    """
    per_event = defaultdict(list)
    per_event_v = defaultdict(list)

    # Build per-event aggregates
    for r in recs:
        jf = r.get('json_file')
        ei = r.get('e_idx')
        if jf is None or pd.isna(jf) or ei is None:
            continue
        cid = r.get('cluster', None)
        if cid is not None:
            per_event[(jf, int(ei))].append(cid)
        v_val = r.get('v', np.nan)
        if not np.isnan(v_val):
            per_event_v[(jf, int(ei))].append(float(v_val))

    # Build index lists per video
    videos = defaultdict(list)
    for (jf, ei) in per_event.keys():
        videos[jf].append(int(ei))
    for (jf, ei) in per_event_v.keys():
        if ei not in videos[jf]:
            videos[jf].append(int(ei))
    for jf in videos:
        videos[jf] = sorted(set(videos[jf]))

    # A global, deterministic fallback cluster-id map for sparse clustering:
    # Use negative ids to avoid colliding with real cluster ids.
    combo_map = {}
    next_combo_id = -1

    results = {}
    for jf, e_idxs in videos.items():
        if not e_idxs:
            results[jf] = dict(
                json_file=jf, n_events=0,
                symbol_seq="", binary_seq="", v_threshold=None,
                symbol_seq_enhanced="", events=[], cycles=[]
            )
            continue

        sym_seq = []            # original token stream (cluster tokens or "_")
        v_list = []             # V means for binary thresholding
        symbol_tokens = []      # full-color symbolic tokens with ↑/↓ and ✦
        events_list = []        # per-event dicts

        for ei in e_idxs:
            cids = per_event.get((jf, ei), [])
            vs = per_event_v.get((jf, ei), [])

            # Representative record for this event
            rec = None
            main_cid = None
            if cids:
                tok = most_common_token(cids, token_map)
                main_cid = Counter(cids).most_common(1)[0][0]
                rec = next((rr for rr in recs
                            if rr.get("json_file") == jf and rr.get("e_idx") == ei and rr.get("cluster") == main_cid), None)
                if rec is None:
                    rec = next((rr for rr in recs
                                if rr.get("json_file") == jf and rr.get("e_idx") == ei), None)
            else:
                tok = "_"
                rec = next((rr for rr in recs
                            if rr.get("json_file") == jf and rr.get("e_idx") == ei), None)

            sym_seq.append(tok)
            v_list.append(np.nan if not vs else float(np.nanmean(vs)))

            # Extract attributes (with safe defaults)
            if rec:
                color_lbl = rec.get("color_label", "Unknown")
                orient    = rec.get("orientation", None)    # "↑" / "↓" / None
                flash_flag= bool(rec.get("flash", False))
                morph     = rec.get("morphology", "unknown")
                st        = rec.get("start_ts", None)
                en        = rec.get("end_ts",   None)
                start_ts  = float(st) if (st is not None and not np.isnan(st)) else None
                end_ts    = float(en) if (en is not None and not np.isnan(en)) else None
                thumb_name= rec.get("thumb")
            else:
                color_lbl = "Unknown"
                orient    = None
                flash_flag= False
                morph     = "unknown"
                start_ts  = None
                end_ts    = None
                # try to find any thumb for this event
                thumb_name= None
                ev_recs = [rr for rr in recs if rr.get("json_file")==jf and rr.get("e_idx")==ei]
                for rr in ev_recs:
                    if rr.get("thumb"):
                        thumb_name = rr["thumb"]
                        break

            # Build enhanced symbol token (full color + optional ↑/↓ + ✦)
            token_str = color_lbl + (orient or "")
            if flash_flag:
                token_str += "✦"
            symbol_tokens.append(token_str)

            # Cluster id for auditor:
            # - Use real cluster id when available
            # - Otherwise create a deterministic fallback based on (color|morph)
            if main_cid is not None:
                cluster_id_for_event = int(main_cid)
            else:
                combo = f"{color_lbl}|{morph}"
                if combo not in combo_map:
                    combo_map[combo] = next_combo_id
                    next_combo_id -= 1
                cluster_id_for_event = combo_map[combo]

            duration = (end_ts - start_ts) if (start_ts is not None and end_ts is not None) else None

            events_list.append({
                "event_index": int(ei),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration": duration,
                "morphology": morph,
                "orientation": orient,
                "flash": flash_flag,
                "color_label": color_lbl,
                "cluster_id": cluster_id_for_event,
                "thumb_obj":  thumb_name
            })

        # Original cluster-token stream
        sym_seq_str = "".join(sym_seq)

        # Enhanced full-color symbolic stream
        sym_seq_enhanced_str = " ".join(symbol_tokens)

        # Binary (V-threshold via median + 0.8*MAD)
        v_arr = np.array(v_list, dtype=float)
        v_med = np.nanmedian(v_arr)
        v_mad = np.nanmedian(np.abs(v_arr - v_med))
        v_thr = v_med + 0.8 * (v_mad if not np.isnan(v_mad) else 0.0)
        bin_seq = [("1" if (not np.isnan(v) and v >= v_thr) else "0") for v in v_arr]
        bin_seq_str = "".join(bin_seq)

                # Make flash more robust using per-video V spikes (OR with existing flags)
        extra = 0.5 * (v_mad if not np.isnan(v_mad) else 0.0)
        for idx, ev in enumerate(events_list):
            vv = v_arr[idx]
            spike = (not np.isnan(vv)) and (vv >= (v_thr + extra))
            if spike and not ev["flash"]:
                ev["flash"] = True
                # add ✦ to enhanced token if not already present
                if not symbol_tokens[idx].endswith("✦"):
                    symbol_tokens[idx] += "✦"
        # refresh enhanced sequence in case we added stars
        sym_seq_enhanced_str = " ".join(symbol_tokens)

        # Cycles: break on >2s gaps OR timestamp resets; always emit a list
        cycles_list = []
        if events_list and any(e.get("start_ts") is not None for e in events_list):
            current = []
            prev_st = events_list[0].get("start_ts")
            for idx, ev in enumerate(events_list):
                st = ev.get("start_ts")
                if (
                    idx > 0 and st is not None and prev_st is not None
                    and ((st - prev_st) > 2.0 or st < prev_st)
                ):
                    cycles_list.append(" ".join(current))
                    current = []
                current.append(symbol_tokens[idx])
                prev_st = st
            if current:
                cycles_list.append(" ".join(current))
        cycles = cycles_list  # always a list (even single cycle)

        results[jf] = {
            "json_file": jf,
            "n_events": len(e_idxs),
            "symbol_seq": sym_seq_str,
            "binary_seq": bin_seq_str,
            "v_threshold": float(v_thr) if not np.isnan(v_thr) else None,
            "symbol_seq_enhanced": sym_seq_enhanced_str,
            "events": events_list,
            "cycles": cycles
        }

    return results

def most_common_token(cids, token_map):
    cnt = Counter([token_map.get(c) for c in cids if c in token_map])
    if not cnt:
        return "_"
    return cnt.most_common(1)[0][0]

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True, help="Path to atlas.csv")
    ap.add_argument("--thumbs", required=True, help="Directory containing thumbs (thumbs_obj)")
    ap.add_argument("--out", required=True, help="Output directory for sequences & codebook")

    ap.add_argument("--min-samples", type=int, default=6, help="DBSCAN min_samples (default 6)")
    ap.add_argument("--eps", type=float, default=0.28, help="DBSCAN eps (default 0.28)")

    ap.add_argument("--min-cluster-size", type=int, default=3,
                    help="Do not assign tokens to clusters smaller than this; they render as 'X' (default 3)")

    ap.add_argument("--w-phash", type=float, default=2.0)
    ap.add_argument("--w-hue", type=float, default=1.0)
    ap.add_argument("--w-feat", type=float, default=1.0)

    ap.add_argument("--gate-hue-deg", type=float, default=14.0)
    ap.add_argument("--gate-area-ratio", type=float, default=0.45)
    ap.add_argument("--gate-solidity", type=float, default=0.20)
    ap.add_argument("--gate-ecc", type=float, default=0.28)
    ap.add_argument("--gate-ar", type=float, default=0.60)

    ap.add_argument("--min-area", type=int, default=50, help="Ignore blobs with area < this (default 50)")

    args = ap.parse_args()

    atlas_csv = Path(args.atlas)
    thumbs_dir = Path(args.thumbs)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    print(f"[load] atlas: {atlas_csv}")
    recs = load_atlas(atlas_csv, thumbs_dir, args.min_area)
    if not recs:
        print("[ERR] atlas had no usable rows.")
        return

    print(f"[cluster] strict-gated DBSCAN (hue-first bucketing). N={len(recs)}")
    recs, by_cluster, feat_scale = cluster_records_strict(recs, args)
    n_clusters = len(by_cluster)
    print(f"[cluster] formed {n_clusters} clusters.")

    print("[tokenize] assign tokens & build codebook…")
    token_map, legend = assign_tokens(by_cluster, min_cluster_size=int(getattr(args, "min_cluster_size", 3)))

    # Build and write codebook
    codebook = dict(
        n_clusters=int(n_clusters),
        weights=dict(w_phash=args.w_phash, w_hue=args.w_hue, w_feat=args.w_feat),
        gates=dict(hue_deg=args.gate_hue_deg,
                   area_ratio=args.gate_area_ratio,
                   solidity=args.gate_solidity,
                   ecc=args.gate_ecc,
                   aspect_ratio=args.gate_ar),
        min_area_px=int(args.min_area),
        legend=legend
    )

    # Add a clusters list for tools that expect codebook["clusters"]
    clusters_list = []
    for tok, meta in legend.items():
        clusters_list.append({
            "cluster_id": int(meta["cluster_id"]),
            "token": tok,
            "count": int(meta.get("count", 0)),
            "prototype": {
                "thumb_obj": meta.get("prototype", {}).get("thumb_obj", "")
            }
        })
    codebook["clusters"] = clusters_list

    codebook_path = out_dir / "symbol_codebook.json"
    write_json(codebook_path, codebook)
    print(f"[OK] codebook → {codebook_path}")

    # Build sequences per video
    print("[sequence] building per-video sequences…")
    seqs = build_sequences(recs, token_map)

    # Per-video JSONs
    for jf, obj in seqs.items():
        # tokens used from original symbol_seq (ignore '_' and spaces)
        used = sorted(set([c for c in obj['symbol_seq'] if c not in ['_', ' ']]))
        vid_legend = {k: legend[k] for k in used if k in legend}

        out = {
            "json_file": jf,
            "n_events": obj["n_events"],
            "legend": vid_legend,
            "symbol_seq": obj["symbol_seq"],
            "symbol_seq_enhanced": obj["symbol_seq_enhanced"],
            "binary_seq": obj["binary_seq"],
            "v_threshold": obj["v_threshold"],
            "events": obj["events"],
            "cycles": obj["cycles"],
            "clustering": {
                "eps": float(args.eps),
                "min_samples": int(args.min_samples),
                "weights": codebook["weights"],
                "gates": codebook["gates"],
                "min_area_px": int(args.min_area),
                "method": "strict-gated (hue-bucketed DBSCAN)"
            }
        }

        stem = Path(jf).name  # e.g., v15044g….json
        out_path = out_dir / f"{stem}.sequence.json"
        write_json(out_path, out)
        print(f"[OK] {stem} → {out_path}")

    print("[DONE]")

if __name__ == "__main__":
    main()
