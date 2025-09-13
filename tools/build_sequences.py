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
    df = pd.read_csv(atlas_csv)
    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    # Expected columns (best-effort): thumb_obj, json_file, e_idx/eid, hsv_mean_h, hsv_mean_s, hsv_mean_v,
    # area, solidity, eccentricity, aspect_ratio, phash64
    # Provide safe fallbacks
    def getcol(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    col_thumb = getcol('thumb_obj', 'thumb', 'thumb_path')
    col_json  = getcol('json_file', 'video', 'source_json')
    col_eidx  = getcol('e_idx', 'event_idx', 'idx', 'eindex', 'e_id', 'eid')
    col_h     = getcol('hsv_mean_h', 'mean_h', 'h')
    col_s     = getcol('hsv_mean_s', 'mean_s', 's')
    col_v     = getcol('hsv_mean_v', 'mean_v', 'v')
    col_area  = getcol('area')
    col_sol   = getcol('solidity')
    col_ecc   = getcol('eccentricity', 'ecc')
    col_ar    = getcol('aspect_ratio', 'ar')
    col_phash = getcol('phash64', 'phash')

    if col_thumb is None or col_json is None or col_eidx is None:
        raise ValueError("atlas.csv must include at least thumb_obj, json_file, and event index (e_idx/eid).")

    # Build records
    recs = []
    missing_needed = 0
    for i, row in df.iterrows():
        thumb = str(row[col_thumb]) if not pd.isna(row[col_thumb]) else None
        json_file = str(row[col_json]) if not pd.isna(row[col_json]) else None
        e_idx = int(row[col_eidx]) if not pd.isna(row[col_eidx]) else None
        h = safe_float(row[col_h]) if col_h else np.nan
        s = safe_float(row[col_s]) if col_s else np.nan
        v = safe_float(row[col_v]) if col_v else np.nan
        area = safe_float(row[col_area]) if col_area else np.nan
        sol  = safe_float(row[col_sol])  if col_sol  else np.nan
        ecc  = safe_float(row[col_ecc])  if col_ecc  else np.nan
        ar   = safe_float(row[col_ar])   if col_ar   else np.nan
        ph   = parse_phash64(row[col_phash]) if col_phash else None

        recs.append(dict(
            idx=i, thumb=thumb, json_file=json_file, e_idx=e_idx,
            h=h, s=s, v=v, area=area, solidity=sol, ecc=ecc, ar=ar,
            phash64=ph,
        ))
        # Need to compute missing basic features?
        if (np.isnan(area) or np.isnan(sol) or np.isnan(ecc) or np.isnan(ar) or ph is None):
            missing_needed += 1

    # Compute missing from thumbs if needed
    if missing_needed > 0 and cv2 is not None:
        for r in recs:
            need_shape = np.isnan(r['area']) or np.isnan(r['solidity']) or np.isnan(r['ecc']) or np.isnan(r['ar'])
            need_phash = (r['phash64'] is None)
            if need_shape or need_phash:
                img = load_img(thumbs_dir, r['thumb']) if r['thumb'] else None
                if need_shape:
                    feats = compute_basic_shape_from_img(img)
                    for k in ['area', 'aspect_ratio', 'solidity', 'eccentricity']:
                        rk = 'ecc' if k == 'eccentricity' else k
                        r[rk] = feats.get(k, r.get(rk, np.nan))
                if need_phash:
                    ph = compute_phash64_from_img(img)
                    r['phash64'] = ph

    # Filter very small area if requested
    if min_area_px and min_area_px > 0:
        recs = [r for r in recs if (np.isnan(r['area']) or r['area'] >= float(min_area_px))]

    # Precompute log_area (safe)
    for r in recs:
        r['log_area'] = np.log(max(1.0, r['area'])) if not np.isnan(r['area']) else np.nan

    return recs

# ---------------------------
# Clustering (strict‑gated)
# ---------------------------

def cluster_records_strict(recs, args):
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

    # Hue-bucketed clustering to keep color-coherent groups and reduce O(N^2)
    # If hue is NaN, put into special bin
    hue_bins = defaultdict(list)
    for r in recs:
        hb = quantize_hue(r['h'], bin_deg=6.0) if not np.isnan(r['h']) else ('nan',)
        hue_bins[hb].append(r)

    # Within each hue bin, build pairwise distances and DBSCAN
    next_cluster_id = 0
    for hb, bucket in hue_bins.items():
        if len(bucket) == 1:
            bucket[0]['cluster'] = next_cluster_id
            next_cluster_id += 1
            continue

        n = len(bucket)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            D[i, i] = 0.0
        for i in range(n):
            ai = bucket[i]
            for j in range(i+1, n):
                aj = bucket[j]
                d = metric.pair_dist(ai, aj, feat_scale)
                D[i, j] = D[j, i] = d

        # DBSCAN on precomputed distances
        db = DBSCAN(eps=float(args.eps), min_samples=int(args.min_samples),
                    metric='precomputed')
        labels = db.fit_predict(D)
        # Assign
        lab_map = {}
        # remap local labels to global cluster ids
        for li in np.unique(labels):
            if li == -1:
                # treat as singleton clusters
                for i, lb in enumerate(labels):
                    if lb == -1:
                        bucket[i]['cluster'] = next_cluster_id
                        next_cluster_id += 1
                continue
            # indices in this local cluster
            idxs = np.where(labels == li)[0]
            # select a medoid
            subD = D[np.ix_(idxs, idxs)]
            m_idx_local = medoid_index(subD)
            m_idx = int(idxs[m_idx_local]) if m_idx_local is not None else int(idxs[0])
            # assign global id
            gid = next_cluster_id
            next_cluster_id += 1
            for k in idxs:
                bucket[k]['cluster'] = gid
            # annotate prototype
            proto = bucket[m_idx]
            for k in idxs:
                bucket[k]['_proto_idx_in_bin'] = m_idx
                bucket[k]['_proto_global'] = gid

    # Collect clusters
    by_cluster = defaultdict(list)
    for r in recs:
        cid = r.get('cluster', None)
        if cid is None:
            # should not happen; put singleton
            cid = next_cluster_id
            next_cluster_id += 1
            r['cluster'] = cid
        by_cluster[cid].append(r)

    # Determine prototype per cluster (global medoid within cluster)
    for cid, rows in by_cluster.items():
        if len(rows) == 1:
            rows[0]['is_prototype'] = True
            continue
        n = len(rows)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i+1, n):
                d = metric.pair_dist(rows[i], rows[j], feat_scale)
                D[i, j] = D[j, i] = d
        m = medoid_index(D)
        for i in range(n):
            rows[i]['is_prototype'] = (i == m)

    return recs, by_cluster, feat_scale

# ---------------------------
# Tokenization & codebook
# ---------------------------

def assign_tokens(by_cluster):
    # Order clusters by size desc, then by prototype hue
    clusters = []
    for cid, rows in by_cluster.items():
        count = len(rows)
        proto = next((r for r in rows if r.get('is_prototype')), rows[0])
        clusters.append((cid, count, proto))
    clusters.sort(key=lambda x: (-x[1], (x[2]['h'] if not np.isnan(x[2]['h']) else 999.0)))

    # Token alphabet A..Z then AA, AB, ...
    def token_gen():
        letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        # A..Z
        for L in letters:
            yield L
        # AA..AZ, BA..BZ, ...
        for a in letters:
            for b in letters:
                yield a + b

    gen = token_gen()
    token_map = {}  # cid -> token
    legend = {}
    for cid, count, proto in clusters:
        tok = next(gen)
        token_map[cid] = tok
        legend[tok] = dict(
            cluster_id=int(cid),
            count=int(count),
            prototype=dict(
                thumb_obj=proto['thumb'],
                json_file=proto['json_file'],
                e_idx=proto['e_idx'],
                h=proto['h'], s=proto['s'], v=proto['v'],
                area=proto['area'], solidity=proto['solidity'],
                ecc=proto['ecc'], ar=proto['ar'],
                phash64=proto['phash64']
            )
        )
    return token_map, legend

# ---------------------------
# Sequence builder
# ---------------------------

def build_sequences(recs, token_map):
    """
    For each video json_file: sort by event index and assign the majority token
    among thumbs that share that event index. Also produce V-based binary seq.
    """
    # Map (json_file, e_idx) -> list of cluster_ids
    per_event = defaultdict(list)
    per_event_v = defaultdict(list)
    for r in recs:
        jf = r['json_file']; ei = r['e_idx']
        if jf is None or pd.isna(jf) or ei is None:
            continue
        cid = r.get('cluster', None)
        if cid is not None:
            per_event[(jf, int(ei))].append(cid)
        if not np.isnan(r['v']):
            per_event_v[(jf, int(ei))].append(float(r['v']))

    # For each video, build sequence
    videos = defaultdict(list)  # jf -> list of event idxs present
    for (jf, ei) in per_event.keys():
        videos[jf].append(int(ei))
    for (jf, ei) in per_event_v.keys():
        if ei not in videos[jf]:
            videos[jf].append(int(ei))
    for jf in videos:
        videos[jf] = sorted(set(videos[jf]))

    results = {}
    for jf, e_idxs in videos.items():
        if not e_idxs:
            results[jf] = dict(json_file=jf, n_events=0, legend={}, symbol_seq="", binary_seq="", v_threshold=None)
            continue

        # majority token per event
        sym_seq = []
        v_list = []
        for ei in e_idxs:
            cids = per_event.get((jf, ei), [])
            if cids:
                tok = most_common_token(cids, token_map)
            else:
                tok = "_"  # underscore for "no symbol for this event"
            sym_seq.append(tok)

            vs = per_event_v.get((jf, ei), [])
            if vs:
                v_list.append(np.mean(vs))
            else:
                v_list.append(np.nan)

        sym_seq_str = "".join(sym_seq)

        # V-based binary (Morse-like) thresholding: robust median + MAD
        v_arr = np.array(v_list, dtype=float)
        v_med = np.nanmedian(v_arr)
        v_mad = np.nanmedian(np.abs(v_arr - v_med))
        # dynamic threshold: median + 0.8*MAD (tunable)
        v_thr = v_med + 0.8 * (v_mad if not np.isnan(v_mad) else 0.0)
        bin_seq = []
        for v in v_arr:
            if np.isnan(v):
                bin_seq.append("0")
            else:
                bin_seq.append("1" if v >= v_thr else "0")
        bin_seq_str = "".join(bin_seq)

        results[jf] = dict(json_file=jf, n_events=len(e_idxs),
                           symbol_seq=sym_seq_str,
                           binary_seq=bin_seq_str,
                           v_threshold=float(v_thr) if not np.isnan(v_thr) else None)
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
    token_map, legend = assign_tokens(by_cluster)

    # Write codebook
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
    codebook_path = out_dir / "symbol_codebook.json"
    write_json(codebook_path, codebook)
    print(f"[OK] codebook → {codebook_path}")

    # Build sequences per video
    print("[sequence] building per-video sequences…")
    seqs = build_sequences(recs, token_map)

    # Also construct per-video legend subset (tokens used in that video)
    # and write per-video sequence json
    for jf, obj in seqs.items():
        # tokens used
        used = sorted(set([c for c in obj['symbol_seq'] if c not in ['_', ' ']]))
        vid_legend = {k: legend[k] for k in used if k in legend}
        out = dict(
            json_file=jf,
            n_events=obj['n_events'],
            legend=vid_legend,
            symbol_seq=obj['symbol_seq'],
            binary_seq=obj['binary_seq'],
            v_threshold=obj['v_threshold'],
            clustering=dict(
                eps=float(args.eps),
                min_samples=int(args.min_samples),
                weights=codebook['weights'],
                gates=codebook['gates'],
                min_area_px=int(args.min_area),
                method="strict-gated (hue-bucketed DBSCAN)"
            )
        )
        stem = Path(jf).name  # e.g., v15044g….json
        out_path = out_dir / f"{stem}.sequence.json"
        write_json(out_path, out)
        print(f"[OK] {stem} → {out_path}")

    print("[DONE]")

if __name__ == "__main__":
    main()
