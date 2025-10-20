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
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Optional; used only if we need to compute missing features
try:
    import cv2
except Exception:
    cv2 = None

from sklearn.cluster import DBSCAN

try:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog  # type: ignore
except Exception:
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]

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
    """Circular hue diff with wraparound (OpenCV HSV: 0..scale).

    The historical implementation assumed that both hue inputs were already
    confined to the same range as ``scale`` (OpenCV style 0..180).  Recent
    pipeline updates store hue in true degrees (0..360).  When the caller still
    used the default 180° scale the previous code would compute ``scale - d``
    for large separations and happily return a *negative* distance
    (e.g. 180 - 200 = -20).  Downstream code compared that value against the
    hue gate, so everything on opposite sides of the color wheel slipped
    through and the clustering step tried to build a 22k×22k distance matrix.

    Normalise the inputs into the [0, scale) interval before measuring the
    delta and take the absolute minimal wraparound distance so we never return
    a negative value regardless of the caller's scale.
    """
    if np.isnan(h1) or np.isnan(h2):
        return np.nan

    s = float(scale)
    if s <= 0 or np.isnan(s):
        return np.nan

    # Bring both hues into the 0..scale range before computing their delta.
    a = float(h1) % s
    b = float(h2) % s
    d = abs(a - b)
    if d > s / 2.0:
        d = s - d
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


def _log(msg: str):
    print(msg, flush=True)


def _mem_mb():
    # optional memory readout without new deps
    import os, gc
    gc.collect()
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss / (1024*1024))
    except Exception:
        return None

# === RADIAL + VACUOLE HELPERS (DROP-IN) ======================================


def _polar_sector_index(cx: float, cy: float, x: float, y: float, n_sectors: int = 8) -> int:
    """
    Angle measured from +X axis, increasing counter-clockwise, mapped to sectors of 360/n.
    We flip Y with a minus sign to keep screen coords consistent.
    """
    ang = math.degrees(math.atan2(-(y - cy), (x - cx))) % 360.0
    w = 360.0 / float(n_sectors)
    return int(ang // w) % n_sectors


def _median_contour_radius(cnt, cx, cy):
    ds = [math.hypot(float(p[0][0]) - cx, float(p[0][1]) - cy) for p in cnt]
    return float(np.median(ds)) if ds else 0.0


def _edge_band_mask_from_contour(shape_hw, cnt, band_px: int) -> np.ndarray:
    """
    Build a thin mask around the contour boundary (band of given thickness).
    """
    h, w = shape_hw
    band_px = max(1, int(band_px))
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=band_px)
    return mask


def _annulus_mask_from_filled(shape_hw, cnt, band_frac: float = 0.12) -> np.ndarray:
    """
    Make an annulus by filling contour then eroding it by a fraction of the approximate radius.
    Useful for ring-like periphery analysis independent of solidity.
    """
    h, w = shape_hw
    fill = np.zeros((h, w), np.uint8)
    cv2.drawContours(fill, [cnt], -1, 255, thickness=cv2.FILLED)

    M = cv2.moments(cnt)
    if M["m00"] <= 0:
        return fill  # fallback
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    r_med = _median_contour_radius(cnt, cx, cy)
    k = max(1, int(band_frac * max(6, r_med)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    inner = cv2.erode(fill, kernel, iterations=1)
    annulus = cv2.subtract(fill, inner)
    return annulus


def compute_radial_and_vacuole_features(
    bgr: np.ndarray,
    cnt: np.ndarray,
    n_sectors: int = 8,
    periphery_band_frac: float = 0.12,
    hough_min_r_frac: float = 0.03,
    hough_max_r_frac: float = 0.12,
    hough_dp: float = 1.2,
    hough_param1: int = 80,
    hough_param2: int = 14,
    hough_min_dist_frac: float = 0.08
) -> dict:
    """
    Compute:
      - orientation8 (ellipse angle -> 8 bins)
      - bright_sector8 (sector with max V-sum in periphery annulus)
      - vacuole detection near periphery via HoughCircles (both bright & dark),
        returning vacuole_count, vacuole_angles_deg, vacuole_sectors8 (list),
        and vacuole_mask8 (bitmask over 8 sectors).

    Returns a dict with fields. Safely handles degenerate contours and small objects.
    """
    out = {
        "orientation8": None,
        "bright_sector8": None,
        "vacuole_count": 0,
        "vacuole_angles_deg": [],
        "vacuole_sectors8": [],
        "vacuole_mask8": 0,
    }
    if bgr is None or cnt is None or len(cnt) < 5:
        return out

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    # Orientation (8 bins) from fitEllipse (OpenCV angle 0..180)
    try:
        ellipse = cv2.fitEllipse(cnt)
        # ellipse = ((cx,cy),(MA,ma),angle)
        angle = float(ellipse[2])  # 0..180 (OpenCV)
        orient8 = int(round(angle / (180.0 / 8.0))) % 8
        out["orientation8"] = orient8
    except Exception:
        out["orientation8"] = None

    # Centroid + reference radius
    M = cv2.moments(cnt)
    if M["m00"] <= 0:
        return out
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    r_med = _median_contour_radius(cnt, cx, cy)
    if r_med <= 1:
        return out

    # Periphery annulus for brightness & vacuoles
    annulus = _annulus_mask_from_filled((h, w), cnt, band_frac=periphery_band_frac)
    if annulus is None or annulus.max() == 0:
        # fallback: thin band along edge
        band_px = max(2, int(0.08 * r_med))
        annulus = _edge_band_mask_from_contour((h, w), cnt, band_px)

    # --- Brightness sector (sum V in each sector of the annulus)
    ys, xs = np.nonzero(annulus)
    if len(xs) >= 3:
        sector_sums = [0.0] * n_sectors
        for py, px in zip(ys, xs):
            sector = _polar_sector_index(cx, cy, px, py, n_sectors=n_sectors)
            sector_sums[sector] += float(V[py, px])
        out["bright_sector8"] = int(np.argmax(sector_sums))
    else:
        out["bright_sector8"] = None

    # --- Vacuole detection near periphery
    # We detect circular blobs in the annulus area in both bright and dark passes.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Radii and min distance relative to object size
    minR = max(2, int(hough_min_r_frac * r_med))
    maxR = max(minR + 1, int(hough_max_r_frac * r_med))
    minDist = max(4, int(hough_min_dist_frac * r_med))

    vac_angles = []
    vac_sectors = []
    mask8 = 0

    def _hough_pass(src_img):
        try:
            circles = cv2.HoughCircles(
                src_img,
                cv2.HOUGH_GRADIENT,
                dp=hough_dp,
                minDist=minDist,
                param1=hough_param1,
                param2=hough_param2,
                minRadius=minR,
                maxRadius=maxR,
            )
        except Exception:
            circles = None
        if circles is None:
            return []
        return circles[0]

    # Two passes: normal and inverted (to catch bright and dark vacuoles)
    candidates = []
    for mat in (blur, (255 - blur)):
        cand = _hough_pass(mat)
        if len(cand):
            candidates.append(cand)
    if len(candidates):
        cand = np.vstack(candidates)
        # Filter: must lie within the periphery annulus and at boundary radius
        # We'll require distance to centroid near r_med, within ±0.35*r_med.
        band_tol = 0.35 * r_med
        ann = annulus > 0
        for (x, y, rr) in cand:
            x, y, rr = float(x), float(y), float(rr)
            # Check inside image
            if not (0 <= int(round(x)) < w and 0 <= int(round(y)) < h):
                continue
            # Must be in the annulus band:
            if not ann[int(round(y)), int(round(x))]:
                continue
            dist = math.hypot(x - cx, y - cy)
            if abs(dist - r_med) > band_tol:
                continue
            ang = math.degrees(math.atan2(-(y - cy), (x - cx))) % 360.0
            sector = _polar_sector_index(cx, cy, x, y, n_sectors=n_sectors)
            vac_angles.append(float(ang))
            vac_sectors.append(int(sector))
            mask8 |= 1 << int(sector)

    # Deduplicate sector hits lightly (optional); here we keep all but report unique sectors in mask.
    out["vacuole_count"] = int(len(vac_angles))
    out["vacuole_angles_deg"] = vac_angles
    out["vacuole_sectors8"] = vac_sectors
    out["vacuole_mask8"] = int(mask8)

    return out

# === END RADIAL + VACUOLE HELPERS ============================================

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
        # --- Gemini-powered semantic gates (run before numeric checks) ---
        def _norm_semantic(value):
            if value is None:
                return ""
            s = str(value).strip().lower()
            if not s or s == "unknown" or s == "nan":
                return ""
            return s

        shape_a = _norm_semantic(a.get('shape_category'))
        shape_b = _norm_semantic(b.get('shape_category'))
        if shape_a and shape_b and shape_a != shape_b:
            return 1.0

        color_a = _norm_semantic(a.get('primary_color'))
        color_b = _norm_semantic(b.get('primary_color'))
        if color_a and color_b and color_a != color_b:
            return 1.0

        # Hard gates
        # Hue
        # Our hue values are expressed in full 0-360° space (either read directly
        # from the atlas or backfilled from OpenCV by multiplying the 0-180°
        # channel by two).  Using the 180° OpenCV scale here caused
        # `circ_hue_delta` to wrap after 90° and even return negative distances
        # for >180° separations, effectively disabling the hue gate for
        # complementary colors.  Use the correct 360° scale so we measure the
        # true angular difference in degrees.
        dh = circ_hue_delta(a['h'], b['h'], scale=360.0)
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
    col_primary = pick("primary_color", "gemini_primary_color", "dominant_color")
    col_shape = pick("shape_category", "gemini_shape", "shape_label")
    col_texture = pick("texture", "gemini_texture", "surface_texture")

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

    # Normalise Gemini semantic labels (if present)
    def _clean_semantic(value: object) -> str:
        if value is None:
            return "unknown"
        try:
            if pd.isna(value):  # type: ignore[attr-defined]
                return "unknown"
        except Exception:
            pass
        s = str(value).strip()
        return s if s else "unknown"

    if col_primary and col_primary in df.columns:
        df[col_primary] = df[col_primary].apply(_clean_semantic)
    if col_shape and col_shape in df.columns:
        df[col_shape] = df[col_shape].apply(_clean_semantic)
    if col_texture and col_texture in df.columns:
        df[col_texture] = df[col_texture].apply(_clean_semantic)

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
            primary_color=_clean_semantic(row[col_primary]) if col_primary else "unknown",
            shape_category=_clean_semantic(row[col_shape]) if col_shape else "unknown",
            texture=_clean_semantic(row[col_texture]) if col_texture else "unknown",
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

    metric_kwargs = {
        "w_phash": args.w_phash,
        "w_hue": args.w_hue,
        "w_feat": args.w_feat,
        "hue_gate_deg": args.gate_hue_deg,
        "area_ratio_gate": args.gate_area_ratio,
        "solidity_gate": args.gate_solidity,
        "ecc_gate": args.gate_ecc,
        "ar_gate": args.gate_ar,
    }
    metric = StrictGatedMetric(**metric_kwargs)

    # Hue buckets
    hue_bins = defaultdict(list)
    for r in recs:
        hb = quantize_hue(r['h'], bin_deg=6.0) if not np.isnan(r['h']) else ('nan',)
        hue_bins[hb].append(r)

    next_cluster_id = 0
    buckets = list(hue_bins.values())
    total_b = len(buckets)
    for bi, bucket in enumerate(buckets, start=1):
        m = len(bucket)
        if m == 0:
            continue
        if m == 1:
            bucket[0]['cluster'] = None
            if args.progress_every:
                mm = _mem_mb()
                mm_s = f" | RAM≈{mm}MB" if mm is not None else ""
                _log(f"[cluster] bucket {bi}/{total_b} size={m}{mm_s}")
                mm = _mem_mb()
                mm_s = f" | RAM≈{mm}MB" if mm is not None else ""
                _log(f"[cluster] bucket {bi}/{total_b} done in 0.0s{mm_s}")
            continue

        t0 = time.time()
        if args.progress_every:
            mm = _mem_mb()
            mm_s = f" | RAM≈{mm}MB" if mm is not None else ""
            _log(f"[cluster] bucket {bi}/{total_b} size={m}{mm_s}")

        D = np.zeros((m, m), dtype=float)

        step = max(1, (m * max(1, args.progress_every)) // 100)

        for i in range(m):
            for j in range(i + 1, m):
                D[i, j] = D[j, i] = metric.pair_dist(bucket[i], bucket[j], feat_scale)

            if args.progress_every and (i % step == 0 or i == m - 1):
                filled_pairs = (i * (i - 1)) // 2
                _log(f"[cluster]   bucket {bi}/{total_b} rows {i + 1}/{m}  pairs≈{filled_pairs}")

        db = DBSCAN(eps=float(args.eps), min_samples=int(args.min_samples), metric='precomputed')
        labels = db.fit_predict(D)

        unique = np.unique(labels)
        for li in unique:
            idxs = np.where(labels == li)[0]
            if li == -1:
                for k in idxs:
                    bucket[k]['cluster'] = None
                continue
            subD = D[np.ix_(idxs, idxs)]
            mloc = np.argmin(subD.sum(axis=1))
            med_k = int(idxs[int(mloc)])
            gid = next_cluster_id
            next_cluster_id += 1
            for k in idxs:
                bucket[k]['cluster'] = gid
            for k in idxs:
                bucket[k]['is_prototype'] = (k == med_k)

        dt = time.time() - t0
        if args.progress_every:
            mm = _mem_mb()
            mm_s = f" | RAM≈{mm}MB" if mm is not None else ""
            _log(f"[cluster] bucket {bi}/{total_b} done in {dt:.1f}s{mm_s}")

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


def extract_event_attributes_with_radial(
    recs,
    thumbs_dir: Path,
    token_map: dict,
    n_sectors: int = 8,
    *,
    enable_radial: bool = True,
    enable_vacuoles: bool = True,
    max_thumb_side: int = 256,
    periphery_band_frac: float = 0.12,
    hough_min_r_frac: float = 0.03,
    hough_max_r_frac: float = 0.12,
    hough_dp: float = 1.2,
    hough_param1: int = 80,
    hough_param2: int = 14,
    hough_min_dist_frac: float = 0.08,
    s_min_for_color: float = 0.12,
    v_min_for_color: float = 0.08,
):
    """
    Aggregate detections per (json_file, e_idx) and compute:
        - majority cluster token
        - color label (prefer existing 'color_label'; else HSV->name; else Unknown)
        - arrow (↑/↓) if present in recs, optional radial/vacuole enrichments.
    Radial/vacuole enrichment (orientation, sector, vacuole_* fields) is only
    computed when the corresponding `enable_radial`/`enable_vacuoles` flags are
    true. Hue-based color inference respects the provided S/V guard rails.

    Returns dict: (jf, e_idx) -> {token, color_label, arrow, orientation8?, ...} 
    """
    by_event = defaultdict(list)
    for r in recs:
        jf = r.get("json_file")
        ei = r.get("e_idx")
        if jf is None or pd.isna(jf) or ei is None:
            continue
        by_event[(jf, int(ei))].append(r)

    out = {}

    # Cache expensive thumbnail analyses so repeated thumbs only pay the cost once
    process_sig = (
        int(n_sectors),
        float(periphery_band_frac),
        float(hough_min_r_frac),
        float(hough_max_r_frac),
        float(hough_dp),
        int(hough_param1),
        int(hough_param2),
        float(hough_min_dist_frac),
        int(max_thumb_side) if isinstance(max_thumb_side, (int, float)) else None,
    )
    thumb_feature_cache = {}

    def analyze_thumb(thumb_name: str):
        key = (thumb_name, process_sig)
        if key in thumb_feature_cache:
            return thumb_feature_cache[key]

        result = None
        if thumbs_dir is None or cv2 is None or not thumb_name:
            thumb_feature_cache[key] = result
            return result

        img = load_img(thumbs_dir, thumb_name)
        if img is None:
            thumb_feature_cache[key] = result
            return result

        if isinstance(max_thumb_side, (int, float)) and max_thumb_side:
            max_side = int(max_thumb_side)
            if max_side > 0:
                h, w = img.shape[:2]
                m = max(h, w)
                if m > max_side:
                    scale = max_side / float(m)
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            2,
        )
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            result = compute_radial_and_vacuole_features(
                img,
                cnt,
                n_sectors=n_sectors,
                periphery_band_frac=periphery_band_frac,
                hough_min_r_frac=hough_min_r_frac,
                hough_max_r_frac=hough_max_r_frac,
                hough_dp=hough_dp,
                hough_param1=hough_param1,
                hough_param2=hough_param2,
                hough_min_dist_frac=hough_min_dist_frac,
            )
        else:
            result = {
                "orientation8": None,
                "bright_sector8": None,
                "vacuole_angles_deg": [],
                "vacuole_sectors8": [],
                "vacuole_mask8": 0,
                "vacuole_count": 0,
            }

        thumb_feature_cache[key] = result
        return result
    for (jf, ei), rows in by_event.items():
        cids = [rr.get("cluster") for rr in rows if rr.get("cluster") is not None]
        token = "_"
        maj_cid = None
        if cids:
            maj_cid = Counter(cids).most_common(1)[0][0]
            toks = [token_map[c] for c in cids if c in token_map]
            if toks:
                token = Counter(toks).most_common(1)[0][0]

        color_lbls = [str(rr.get("color_label")) for rr in rows if rr.get("color_label")]
        if color_lbls:
            color_label = Counter(color_lbls).most_common(1)[0][0]
        else:
            valid_hues = []
            for rr in rows:
                h_val = safe_float(rr.get("h"), np.nan)
                if np.isnan(h_val):
                    continue
                s_val = safe_float(rr.get("s"), np.nan)
                v_val = safe_float(rr.get("v"), np.nan)
                if not np.isnan(s_val) and s_val < s_min_for_color:
                    continue
                if not np.isnan(v_val) and v_val < v_min_for_color:
                    continue
                valid_hues.append(h_val)
            color_label = "Unknown"
            if valid_hues:
              color_label = "Unknown"
              if valid_hues:
                  hmed = float(np.median(valid_hues))
                  if 35.0 <= hmed < 70.0:
                      color_label = "Yellow"
                  elif 80.0 <= hmed < 160.0:
                      color_label = "Green"
                  elif 200.0 <= hmed < 260.0:
                      color_label = "Blue"
                  elif hmed >= 300.0 or hmed < 10.0:
                      color_label = "Pink/Magenta"

        thumbs = [rr.get("thumb") for rr in rows if rr.get("thumb")]
        thumb = Counter(thumbs).most_common(1)[0][0] if thumbs else None

        arrow = None
        for rr in rows:
            orient_str = str(rr.get("orientation", ""))
            if orient_str.endswith("↑"):
                arrow = "↑"
                break
            if orient_str.endswith("↓"):
                arrow = "↓"
                break
        
        flash_vals = [bool(rr.get("flash")) for rr in rows]
        flash_flag = any(flash_vals)

        orient8 = None
        bright_sector8 = None
        vac_angles = []
        vac_sectors = []
        vac_mask8 = 0
        vac_count = 0
        want_radial = enable_radial or enable_vacuoles
        if thumb and want_radial:
            feats = analyze_thumb(thumb)
            if feats:
                if enable_radial:
                    orient8 = feats.get("orientation8")
                    bright_sector8 = feats.get("bright_sector8")
                if enable_vacuoles:
                    vac_angles = feats.get("vacuole_angles_deg", [])
                    vac_sectors = feats.get("vacuole_sectors8", [])
                    vac_mask8 = feats.get("vacuole_mask8", 0)
                    vac_count = feats.get("vacuole_count", 0)

        sector_idx = None if bright_sector8 is None else int(bright_sector8)
        if sector_idx is None:
            sector_label = "sector_unknown"
        else:
            sector_label = f"sector_{sector_idx}_of_{n_sectors}"

        out[(jf, ei)] = dict(
            token=token,
            cluster_id=None if maj_cid is None else int(maj_cid),
            color_label=color_label,
            arrow=arrow,
            flash=flash_flag,  
            orientation8=None if orient8 is None else int(orient8),
            sector8=sector_idx,
            sector_label=sector_label,
            vacuole_count=int(vac_count),
            vacuole_angles_deg=[float(a) for a in vac_angles],
            vacuole_sectors8=[int(s) for s in vac_sectors],
            vacuole_mask8=int(vac_mask8),
            thumb_obj=thumb,
        )

    return out


# ====== SEQUENCE BUILDER (DROP-IN) ============================================
def build_sequences(recs, token_map, thumbs_dir: Path, args):
    """
    Build per-video sequences with enriched per-event attributes:
      - token (majority), cluster_id (majority)
      - color_label (full names)
      - arrow (↑/↓ when seen in prior pipeline)
      - orientation8 (0..7), sector8 (0..7)
      - vacuole_* (count, angles, sectors, mask8)
      - enhanced_token for human-readable stream
    Also produce:
      - symbol_seq (classic token stream),
      - binary_seq from V,
      - cycles split by time gaps/resets.
    """
    from collections import defaultdict
    import numpy as np

    # brightness list for binary thresholding
    per_event_v = defaultdict(list)
    for r in recs:
        jf = r.get('json_file'); ei = r.get('e_idx')
        if jf is None or pd.isna(jf) or ei is None:
            continue
        if not np.isnan(r.get('v', np.nan)):
            per_event_v[(jf, int(ei))].append(float(r['v']))

    # video -> sorted event indices
    videos = defaultdict(set)
    for r in recs:
        jf = r.get('json_file'); ei = r.get('e_idx')
        if jf is None or pd.isna(jf) or ei is None:
            continue
        videos[jf].add(int(ei))
    for (jf, ei) in per_event_v.keys():
        videos[jf].add(int(ei))
    videos = {jf: sorted(list(eis)) for jf, eis in videos.items()}

    # Attributes per event
    attrs = extract_event_attributes_with_radial(
        recs, thumbs_dir, token_map,
        n_sectors=args.sectors,
        enable_radial=args.enable_radial,
        enable_vacuoles=args.enable_vacuoles,
        max_thumb_side=args.max_thumb_side,
        periphery_band_frac=args.radial_band_frac,
        hough_min_r_frac=args.hough_minr_frac,
        hough_max_r_frac=args.hough_maxr_frac,
        hough_dp=args.hough_dp,
        hough_param1=args.hough_param1,
        hough_param2=args.hough_param2,
        hough_min_dist_frac=args.hough_mindist_frac,
        s_min_for_color=args.s_min_for_color,
        v_min_for_color=args.v_min_for_color
    )

    results = {}
    for jf, e_idxs in videos.items():
        if not e_idxs:
            results[jf] = dict(
                json_file=jf, n_events=0, legend={}, symbol_seq="",
                binary_seq="", v_threshold=None, symbol_seq_enhanced="",
                events=[], cycles=[]
            )
            continue

        raw_tokens, v_vals, events_list, symbol_tokens = [], [], [], []
        for ei in e_idxs:
            a = attrs.get((jf, ei), None)
            tok = a["token"] if a else "_"
            raw_tokens.append(tok)

            # V for binary threshold
            vv = per_event_v.get((jf, ei), [])
            v_vals.append(np.nan if not vv else float(np.nanmean(vv)))

            # Best-effort timestamps
            rr = next((rr for rr in recs if rr.get("json_file")==jf and rr.get("e_idx")==ei), None)
            st = rr.get("start_ts") if rr else None
            en = rr.get("end_ts") if rr else None
            start_ts = float(st) if (st is not None and not np.isnan(st)) else None
            end_ts   = float(en) if (en is not None and not np.isnan(en)) else None
            duration = (end_ts - start_ts) if (start_ts is not None and end_ts is not None) else None

            if a:
                parts = []
                clr = a.get("color_label", "Unknown")
                arr = a.get("arrow") or ""
                if (not arr) and rr:
                    orient_str = str(rr.get("orientation") or "")
                    if orient_str.endswith("↑"):
                        arr = "↑"
                    elif orient_str.endswith("↓"):
                        arr = "↓"
                flash_flag = bool(a.get("flash"))
                base_tok = clr + arr
                if flash_flag and not base_tok.endswith("✦"):
                    base_tok += "✦"
                parts.append(base_tok)
                if a.get("orientation8") is not None:
                    parts.append(f"o{int(a['orientation8'])}")
                if a.get("sector8") is not None:
                    parts.append(f"s{int(a['sector8'])}")
                vc = int(a.get("vacuole_count", 0))
                if vc > 0:
                    vsec = a.get("vacuole_sectors8", [])
                    if vsec:
                        uniq = sorted(set([int(x) for x in vsec]))
                        parts.append("V[" + ",".join(str(x) for x in uniq) + "]")
                    else:
                        parts.append("Vx")
                else:
                    parts.append("V0")
                enhanced_display = " ".join(parts)
              
                thumb_name = a.get("thumb_obj", None)
                if not thumb_name and rr:
                    thumb_name = rr.get("thumb")
                events_list.append({
                    "event_index": int(ei),
                    "start_ts": start_ts, "end_ts": end_ts, "duration": duration,
                    "morphology": rr.get("morphology","unknown") if rr else "unknown",
                    "color_label": clr,
                    "arrow": arr,
                    "orientation": arr or None,
                    "flash": flash_flag,
                    "orientation8": a.get("orientation8", None),
                    "sector8": a.get("sector8", None),
                    "vacuole_count": vc,
                    "vacuole_angles_deg": a.get("vacuole_angles_deg", []),
                    "vacuole_sectors8": a.get("vacuole_sectors8", []),
                    "vacuole_mask8": a.get("vacuole_mask8", 0),
                    "cluster_id": a.get("cluster_id", None),
                    "thumb_obj": thumb_name,
                    "enhanced_token": enhanced_display
                })
                symbol_tokens.append(enhanced_display)
            else:
                flash_flag = bool(rr.get("flash", False)) if rr else False
                base_tok = "Unknown"
                if flash_flag:
                    base_tok += "✦"
                enhanced_display = f"{base_tok} V0"
                thumb_name = rr.get("thumb") if rr else None
                events_list.append({
                    "event_index": int(ei),
                    "start_ts": start_ts, "end_ts": end_ts, "duration": duration,
                    "morphology": rr.get("morphology","unknown") if rr else "unknown",
                    "color_label": "Unknown", "arrow": "",
                    "orientation": None,
                    "flash": flash_flag,
                    "orientation8": None, "sector8": None,
                    "vacuole_count": 0, "vacuole_angles_deg": [],
                    "vacuole_sectors8": [], "vacuole_mask8": 0,
                    "cluster_id": None, "thumb_obj": thumb_name,
                    "enhanced_token": enhanced_display
                })
                symbol_tokens.append(enhanced_display)

        symbol_seq = "".join(raw_tokens)
        v_arr = np.array(v_vals, dtype=float)
        v_med = np.nanmedian(v_arr)
        v_mad = np.nanmedian(np.abs(v_arr - v_med))
        v_thr = v_med + 0.8 * (v_mad if not np.isnan(v_mad) else 0.0)
        binary_seq = "".join(("1" if (not np.isnan(v) and v >= v_thr) else "0") for v in v_arr)
        extra = 0.5 * (v_mad if not np.isnan(v_mad) else 0.0)
        for idx, ev in enumerate(events_list):
            vv = v_arr[idx]
            spike = (not np.isnan(vv)) and (vv >= (v_thr + extra))
            if spike and not ev.get("flash", False):
                ev["flash"] = True
                parts = symbol_tokens[idx].split(" ") if symbol_tokens[idx] else []
                if parts:
                    if not parts[0].endswith("✦"):
                        parts[0] = parts[0] + "✦"
                    updated = " ".join(parts)
                    symbol_tokens[idx] = updated
                    ev["enhanced_token"] = updated
        symbol_seq_enhanced = " ".join(symbol_tokens)

        # cycles by >2s gap or time reset
        cycles_list = []
        if events_list and any(e.get("start_ts") is not None for e in events_list):
            current = []
            prev_st = events_list[0].get("start_ts")
            for idx, ev in enumerate(events_list):
                st = ev.get("start_ts")
                if idx > 0 and st is not None and prev_st is not None and ((st - prev_st) > 2.0 or st < prev_st):
                    cycles_list.append(" ".join(current))
                    current = []
                current.append(ev["enhanced_token"])
                prev_st = st
            if current: cycles_list.append(" ".join(current))

        results[jf] = {
            "json_file": jf,
            "n_events": len(e_idxs),
            "symbol_seq": symbol_seq,
            "binary_seq": binary_seq,
            "v_threshold": float(v_thr) if not np.isnan(v_thr) else None,
            "symbol_seq_enhanced": symbol_seq_enhanced,
            "events": events_list,
            "cycles": cycles_list
        }
    return results
# ==============================================================================



def most_common_token(cids, token_map):
    cnt = Counter([token_map.get(c) for c in cids if c in token_map])
    if not cnt:
        return "_"
    return cnt.most_common(1)[0][0]

# ---------------------------
# Main
# ---------------------------

# ====== MAIN (DROP-IN) ========================================================
def _create_dialog_root() -> "tk.Tk":  # type: ignore[name-defined]
    if tk is None or filedialog is None:  # type: ignore[truthy-function]
        raise RuntimeError(
            "tkinter is unavailable. Supply the required paths using command-line arguments."
        )

    try:
        root = tk.Tk()  # type: ignore[call-arg]
        root.withdraw()
        return root
    except Exception as exc:
        raise RuntimeError(
            "Unable to launch a file-selection dialog. Provide the paths explicitly via CLI options."
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


def _pick_out_dir() -> Optional[Path]:
    root = _create_dialog_root()
    try:
        dirname = filedialog.askdirectory(  # type: ignore[union-attr]
            title="Select output directory for sequences",
        )
    finally:
        root.destroy()
    return Path(dirname).expanduser() if dirname else None


def _resolve_cli_path(
    value: Optional[str],
    *,
    picker: Callable[[], Optional[Path]],
    descriptor: str,
) -> Path:
    if value:
        return Path(value).expanduser()

    try:
        selected = picker()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not selected:
        raise SystemExit(f"No {descriptor} selected; aborting.")

    return selected


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", help="Path to atlas.csv")
    ap.add_argument("--thumbs", help="Directory containing thumbs (thumbs_obj)")
    ap.add_argument("--out", help="Output directory for sequences & codebook")

    # clustering/tokens (kept)
    ap.add_argument("--min-samples", type=int, default=6, help="DBSCAN min_samples")
    ap.add_argument("--eps", type=float, default=0.28, help="DBSCAN eps")
    ap.add_argument("--min-cluster-size", type=int, default=3,
                    help="Clusters smaller than this render as 'X'")

    ap.add_argument("--w-phash", type=float, default=2.0)
    ap.add_argument("--w-hue", type=float, default=1.0)
    ap.add_argument("--w-feat", type=float, default=1.0)

    ap.add_argument("--gate-hue-deg", type=float, default=14.0)
    ap.add_argument("--gate-area-ratio", type=float, default=0.45)
    ap.add_argument("--gate-solidity", type=float, default=0.20)
    ap.add_argument("--gate-ecc", type=float, default=0.28)
    ap.add_argument("--gate-ar", type=float, default=0.60)

    ap.add_argument("--min-area", type=int, default=50, help="Ignore blobs with area < this")

    ap.add_argument(
        "--progress-every", type=int, default=10,
        help="Percent step for per-bucket progress logs (0 disables)."
    )

    # NEW: radial/vacuole controls
    ap.add_argument("--sectors", type=int, default=8)
    ap.add_argument("--enable-radial", action="store_true", help="Compute orientation8/sector8")
    ap.add_argument("--enable-vacuoles", action="store_true", help="Detect vacuoles on periphery")
    ap.add_argument("--max-thumb-side", type=int, default=256)
    ap.add_argument("--radial-band-frac", type=float, default=0.12)

    ap.add_argument("--hough-minr-frac", type=float, default=0.03)
    ap.add_argument("--hough-maxr-frac", type=float, default=0.12)
    ap.add_argument("--hough-dp", type=float, default=1.2)
    ap.add_argument("--hough-param1", type=int, default=80)
    ap.add_argument("--hough-param2", type=int, default=14)
    ap.add_argument("--hough-mindist-frac", type=float, default=0.08)

    ap.add_argument("--s-min-for-color", type=float, default=0.12)
    ap.add_argument("--v-min-for-color", type=float, default=0.08)

    args = ap.parse_args()

    atlas_csv = _resolve_cli_path(
        args.atlas,
        picker=_pick_atlas_csv,
        descriptor="atlas CSV (--atlas)",
    )
    thumbs_dir = _resolve_cli_path(
        args.thumbs,
        picker=_pick_thumbs_dir,
        descriptor="thumbnail directory (--thumbs)",
    )
    out_dir = _resolve_cli_path(
        args.out,
        picker=_pick_out_dir,
        descriptor="output directory (--out)",
    )
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

    # Build codebook
    codebook = dict(
        n_clusters=int(n_clusters),
        weights=dict(w_phash=args.w_phash, w_hue=args.w_hue, w_feat=args.w_feat),
        gates=dict(hue_deg=args.gate_hue_deg,
                   area_ratio=args.gate_area_ratio,
                   solidity=args.gate_solidity,
                   ecc=args.gate_ecc,
                   aspect_ratio=args.gate_ar),
        min_area_px=int(args.min_area),
        radial=dict(
            sectors=int(args.sectors),
            enable_radial=bool(args.enable_radial),
            enable_vacuoles=bool(args.enable_vacuoles),
            band_frac=float(args.radial_band_frac),
            hough=dict(
                minr_frac=float(args.hough_minr_frac),
                maxr_frac=float(args.hough_maxr_frac),
                dp=float(args.hough_dp),
                p1=int(args.hough_param1),
                p2=int(args.hough_param2),
                mindist_frac=float(args.hough_mindist_frac)
            ),
            color_guards=dict(s_min=float(args.s_min_for_color), v_min=float(args.v_min_for_color)),
            max_thumb_side=int(args.max_thumb_side)
        ),
        legend=legend
    )

    # Provide 'clusters' list for tools that expect it
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

    # Build sequences per video (NOTE: pass thumbs_dir and args)
    print("[sequence] building per-video sequences…")
    seqs = build_sequences(recs, token_map, thumbs_dir, args)

    _log(f"[write] assembling sequences… K={len(by_cluster)} clusters")

    # Per-video JSONs
    total_seqs = len(seqs)
    for idx, (jf, obj) in enumerate(seqs.items(), start=1):
        used = sorted(set([c for c in obj['symbol_seq'] if c not in ['_', ' ']]))
        vid_legend = {tok: legend[tok] for tok in used if tok in legend}

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

        stem = Path(jf).name
        out_path = out_dir / f"{stem}.sequence.json"
        _log(f"[write] {idx}/{total_seqs} {stem} events={obj['n_events']} → {out_path.name}")
        write_json(out_path, out)
        print(f"[OK] {stem} → {out_path}")

    print("[DONE]")

if __name__ == "__main__":
    main()
# ==============================================================================

