diff --git a/tools/remove_tiktok_thumbs.py b/tools/remove_tiktok_thumbs.py
index 8480ba93dcb6135d45f7099f9253d11898394a65..6a354039efd0b68429ab1cf3d902c8cb53ac4c45 100644
--- a/tools/remove_tiktok_thumbs.py
+++ b/tools/remove_tiktok_thumbs.py
@@ -1,77 +1,210 @@
 ﻿#!/usr/bin/env python3
 # tools/remove_tiktok_thumbs.py
 #
 # Default run:
 #   python tools/remove_tiktok_thumbs.py thumbs thumbs_obj --aggressive
 # This blacks out full tiles with the tuned thresholds (min-red=0.002, min-cyan=0.002, rc-min=0.0008).
 
 import argparse, os, json, cv2, numpy as np
 from pathlib import Path
-from typing import Tuple
+from typing import Optional, Tuple
 
 # ---------------------------- HSV color ranges ----------------------------
 # OpenCV HSV: H in [0,180)
 RED_1  = (np.array([  0, 50, 50]), np.array([ 10,255,255]))
 RED_2  = (np.array([160, 50, 50]), np.array([180,255,255]))
 CYAN   = (np.array([ 80, 50, 50]), np.array([100,255,255]))
 
 # --------------------------- Helper detectors ----------------------------
 
 def red_cyan_ratios(img: np.ndarray) -> Tuple[float, float]:
     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
     total = hsv.shape[0] * hsv.shape[1]
     m1 = cv2.inRange(hsv, RED_1[0], RED_1[1])
     m2 = cv2.inRange(hsv, RED_2[0], RED_2[1])
     mr = cv2.bitwise_or(m1, m2)
     mc = cv2.inRange(hsv, CYAN[0], CYAN[1])
     return (float(np.count_nonzero(mr)) / max(total, 1),
             float(np.count_nonzero(mc)) / max(total, 1))
 
 def corner_mask(img_shape, margin: float = 0.22):
     """Return 4 binary masks for TL, TR, BL, BR corner windows."""
     h, w = img_shape[:2]
     mW, mH = int(w * margin), int(h * margin)
     masks = []
     for (x0, y0) in [(0,0), (w - mW, 0), (0, h - mH), (w - mW, h - mH)]:
         m = np.zeros((h, w), np.uint8)
         m[y0:y0 + mH, x0:x0 + mW] = 255
         masks.append(m)
     return masks
 
 def rc_overlap_ratio(hsv: np.ndarray, kernel_size: int = 5, dil_iters: int = 2) -> float:
     """Fraction of pixels where dilated red overlaps cyan — a strong TT signature."""
     H, W = hsv.shape[:2]
     m1 = cv2.inRange(hsv, RED_1[0], RED_1[1])
     m2 = cv2.inRange(hsv, RED_2[0], RED_2[1])
     m_red  = cv2.bitwise_or(m1, m2)
     m_cyan = cv2.inRange(hsv, CYAN[0], CYAN[1])
     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
     red_dil = cv2.dilate(m_red, k, iterations=dil_iters)
     overlap = cv2.bitwise_and(red_dil, m_cyan)
     return float(np.count_nonzero(overlap)) / float(H * W)
 
+
+def expand_box(box: Tuple[int, int, int, int], pad: float, width: int, height: int) -> Tuple[int, int, int, int]:
+    """Pad a bounding box by `pad` proportion while keeping it inside the image."""
+    if pad <= 0:
+        return box
+    x0, y0, x1, y1 = box
+    w = max(0, x1 - x0 + 1)
+    h = max(0, y1 - y0 + 1)
+    if w == 0 or h == 0:
+        return box
+    dw, dh = int(round(w * pad)), int(round(h * pad))
+    x0 = max(0, x0 - dw)
+    y0 = max(0, y0 - dh)
+    x1 = min(width - 1, x1 + dw)
+    y1 = min(height - 1, y1 + dh)
+    return x0, y0, x1, y1
+
+
+def detect_notification_badge(
+    img: np.ndarray,
+    min_area_ratio: float = 0.0005,
+    max_area_ratio: float = 0.05,
+    min_fill_ratio: float = 0.4,
+    min_white_ratio: float = 0.08,
+) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
+    """Detect red notification badges with white numerals (e.g., TikTok friends counter)."""
+
+    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
+    H, W = hsv.shape[:2]
+
+    # Bright pink/red bubble
+    badge_mask_1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
+    badge_mask_2 = cv2.inRange(hsv, np.array([160, 120, 120]), np.array([180, 255, 255]))
+    badge_mask = cv2.bitwise_or(badge_mask_1, badge_mask_2)
+
+    if np.count_nonzero(badge_mask) == 0:
+        return False, None
+
+    badge_mask = cv2.medianBlur(badge_mask, 3)
+    cnts, _ = cv2.findContours(badge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
+
+    for cnt in cnts:
+        area = float(cv2.contourArea(cnt))
+        if area <= 0:
+            continue
+        area_ratio = area / float(H * W)
+        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
+            continue
+
+        x, y, w, h = cv2.boundingRect(cnt)
+        if w == 0 or h == 0:
+            continue
+
+        aspect = w / float(h)
+        if aspect < 0.5 or aspect > 1.8:
+            continue
+
+        rect_area = float(w * h)
+        fill_ratio = area / rect_area
+        if fill_ratio < min_fill_ratio:
+            continue
+
+        roi_hsv = hsv[y : y + h, x : x + w]
+        white_mask = cv2.inRange(roi_hsv, np.array([0, 0, 200]), np.array([180, 60, 255]))
+        white_ratio = float(np.count_nonzero(white_mask)) / rect_area
+        if white_ratio < min_white_ratio:
+            continue
+
+        return True, (x, y, x + w - 1, y + h - 1)
+
+    return False, None
+
+
+def detect_golden_medallion(
+    img: np.ndarray,
+    min_area_ratio: float = 0.0002,
+    max_area_ratio: float = 0.04,
+    min_fill_ratio: float = 0.25,
+    min_dark_ratio: float = 0.08,
+    min_circularity: float = 0.45,
+) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
+    """Detect circular gold medallion avatars resembling the provided sample."""
+
+    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
+    H, W = hsv.shape[:2]
+
+    gold_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([45, 255, 255]))
+    if np.count_nonzero(gold_mask) == 0:
+        return False, None
+
+    gold_mask = cv2.GaussianBlur(gold_mask, (5, 5), 0)
+    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE,
+                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
+
+    cnts, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
+    for cnt in cnts:
+        area = float(cv2.contourArea(cnt))
+        if area <= 0:
+            continue
+        area_ratio = area / float(H * W)
+        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
+            continue
+
+        peri = cv2.arcLength(cnt, True)
+        if peri <= 0:
+            continue
+        circularity = 4.0 * np.pi * area / (peri * peri)
+        if circularity < min_circularity:
+            continue
+
+        x, y, w, h = cv2.boundingRect(cnt)
+        if w == 0 or h == 0:
+            continue
+
+        aspect = w / float(h)
+        if aspect < 0.75 or aspect > 1.3:
+            continue
+
+        rect_area = float(w * h)
+        fill_ratio = area / rect_area
+        if fill_ratio < min_fill_ratio:
+            continue
+
+        roi_hsv = hsv[y : y + h, x : x + w]
+        dark_mask = cv2.inRange(roi_hsv, np.array([0, 0, 0]), np.array([180, 255, 120]))
+        dark_ratio = float(np.count_nonzero(dark_mask)) / rect_area
+        if dark_ratio < min_dark_ratio:
+            continue
+
+        return True, (x, y, x + w - 1, y + h - 1)
+
+    return False, None
+
 def detect_tiktok_aggressive(img: np.ndarray,
                              min_red: float = 0.004,
                              min_cyan: float = 0.004,
                              rc_min: float = 0.0012) -> bool:
     """
     Aggressive global detector:
     triggers if (red_ratio OR cyan_ratio) is non-trivial AND red/cyan overlap exists.
     """
     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
     H, W = hsv.shape[:2]
     m1 = cv2.inRange(hsv, RED_1[0], RED_1[1])
     m2 = cv2.inRange(hsv, RED_2[0], RED_2[1])
     m_red  = cv2.bitwise_or(m1, m2)
     m_cyan = cv2.inRange(hsv, CYAN[0], CYAN[1])
     red_ratio  = np.count_nonzero(m_red)  / (H * W)
     cyan_ratio = np.count_nonzero(m_cyan) / (H * W)
     if (red_ratio >= min_red or cyan_ratio >= min_cyan):
         if rc_overlap_ratio(hsv, kernel_size=5, dil_iters=2) >= rc_min:
             return True
     return False
 
 # ---------------------------- Standard detector --------------------------
 
 def detect_tiktok(img: np.ndarray, min_red=0.01, min_cyan=0.05,
                   corner_only=True, margin=0.22, pad=0.10):
diff --git a/tools/remove_tiktok_thumbs.py b/tools/remove_tiktok_thumbs.py
index 8480ba93dcb6135d45f7099f9253d11898394a65..6a354039efd0b68429ab1cf3d902c8cb53ac4c45 100644
--- a/tools/remove_tiktok_thumbs.py
+++ b/tools/remove_tiktok_thumbs.py
@@ -113,50 +246,68 @@ def detect_tiktok(img: np.ndarray, min_red=0.01, min_cyan=0.05,
     if red_ratio > min_red and cyn_ratio > min_cyan:
         comb = cv2.bitwise_or(m_red, m_cyan)
         cnts, _ = cv2.findContours(comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         if cnts:
             x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
             dw, dh = int(w * pad), int(h * pad)
             return True, (max(0, x - dw), max(0, y - dh), min(W-1, x + w + dw), min(H-1, y + h + dh))
     return False, None
 
 # ------------------------------- Worker ----------------------------------
 
 def process_file(fp: Path, mode: str,
                  min_red: float, min_cyan: float,
                  corner_only: bool, margin: float, pad: float,
                  aggressive: bool, rc_min: float):
     img = cv2.imread(str(fp))
     if img is None:
         return False, "unreadable"
     H, W = img.shape[:2]
 
     # Standard detector: corner → global fallback
     is_logo, box = detect_tiktok(img, min_red, min_cyan, corner_only, margin, pad=pad)
     if not is_logo and corner_only:
         is_logo, box = detect_tiktok(img, min_red, min_cyan, False, margin, pad=pad)
 
+    if not is_logo:
+        badge_hit, badge_box = detect_notification_badge(img)
+        if badge_hit:
+            is_logo = True
+            if badge_box is not None:
+                box = expand_box(badge_box, pad, W, H)
+            else:
+                box = None
+
+    if not is_logo:
+        medallion_hit, medallion_box = detect_golden_medallion(img)
+        if medallion_hit:
+            is_logo = True
+            if medallion_box is not None:
+                box = expand_box(medallion_box, pad, W, H)
+            else:
+                box = None
+
     # Aggressive fallback (global red–cyan overlap)
     if not is_logo and aggressive:
         if detect_tiktok_aggressive(img, min_red=min_red, min_cyan=min_cyan, rc_min=rc_min):
             is_logo, box = True, None
 
     if not is_logo:
         return False, "ok"
 
     if mode == "delete":
         fp.unlink(missing_ok=True)
         return True, "deleted"
 
     if mode == "black":
         # FULL-TILE BLACKOUT
         black = np.zeros((H, W, 3), np.uint8)
         cv2.imwrite(str(fp), black)
         return True, "black"
 
     if mode == "mask":
         # Rectangle fill (kept for completeness)
         if box is None:
             x0, y0, x1, y1 = 0, 0, int(W * margin), int(H * margin)
         else:
             x0, y0, x1, y1 = box
         cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
diff --git a/tools/remove_tiktok_thumbs.py b/tools/remove_tiktok_thumbs.py
index 8480ba93dcb6135d45f7099f9253d11898394a65..6a354039efd0b68429ab1cf3d902c8cb53ac4c45 100644
--- a/tools/remove_tiktok_thumbs.py
+++ b/tools/remove_tiktok_thumbs.py
@@ -188,47 +339,55 @@ def main():
     p.add_argument("--rc-min", type=float, default=0.0008,
                    help="min red–cyan overlap fraction for aggressive detector (default tuned to 0.0008)")
     p.add_argument("--log", default=None, help="write JSON action log")
     args = p.parse_args()
 
     actions = []
     for root in args.paths:
         for dirpath, _, files in os.walk(root):
             for name in files:
                 if name.lower().endswith((".jpg",".jpeg",".png",".bmp",".gif")):
                     fp = Path(dirpath) / name
 
                     # 'move' mode: decide, then move
                     if args.mode == "move":
                         img = cv2.imread(str(fp))
                         if img is None:
                             continue
                         hit, _ = detect_tiktok(img, args.min_red, args.min_cyan,
                                                args.corner_only, args.margin, pad=args.pad)
                         if not hit and args.corner_only:
                             hit, _ = detect_tiktok(img, args.min_red, args.min_cyan,
                                                    False, args.margin, pad=args.pad)
                         if not hit and args.aggressive:
                             hit = detect_tiktok_aggressive(img, min_red=args.min_red,
                                                            min_cyan=args.min_cyan, rc_min=args.rc_min)
+                        if not hit:
+                            badge_hit, _ = detect_notification_badge(img)
+                            if badge_hit:
+                                hit = True
+                        if not hit:
+                            medallion_hit, _ = detect_golden_medallion(img)
+                            if medallion_hit:
+                                hit = True
                         if hit and args.move_to:
                             os.makedirs(args.move_to, exist_ok=True)
                             dest = Path(args.move_to) / name
                             if dest.exists():
                                 dest = dest.with_stem(dest.stem + "_dup")
                             fp.replace(dest)
                             actions.append({"file": str(fp), "action": "move", "dest": str(dest)})
                         continue
 
                     changed, note = process_file(fp, args.mode, args.min_red, args.min_cyan,
                                                  args.corner_only, args.margin, args.pad,
                                                  args.aggressive, args.rc_min)
                     if changed:
                         actions.append({"file": str(fp), "action": args.mode, "note": note})
 
     if args.log:
         with open(args.log, "w", encoding="utf-8") as jf:
             json.dump(actions, jf, ensure_ascii=False, indent=2)
     print(f"Done. {len(actions)} files modified.")
 
 if __name__ == "__main__":
     main()
