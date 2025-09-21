diff --git a/tools/remove_tiktok_thumbs.py b/tools/remove_tiktok_thumbs.py
index 8480ba93dcb6135d45f7099f9253d11898394a65..eed961450ffbcc251b1faf674b653e8319e2f3d6 100644
--- a/tools/remove_tiktok_thumbs.py
+++ b/tools/remove_tiktok_thumbs.py
@@ -1,77 +1,141 @@
 ﻿#!/usr/bin/env python3
 # tools/remove_tiktok_thumbs.py
 #
 # Default run:
 #   python tools/remove_tiktok_thumbs.py thumbs thumbs_obj --aggressive
 # This blacks out full tiles with the tuned thresholds (min-red=0.002, min-cyan=0.002, rc-min=0.0008).
 
 import argparse, os, json, cv2, numpy as np
 from pathlib import Path
 from typing import Tuple
 
 # ---------------------------- HSV color ranges ----------------------------
 # OpenCV HSV: H in [0,180)
 RED_1  = (np.array([  0, 50, 50]), np.array([ 10,255,255]))
 RED_2  = (np.array([160, 50, 50]), np.array([180,255,255]))
 CYAN   = (np.array([ 80, 50, 50]), np.array([100,255,255]))
+WHITE  = (np.array([  0,  0,200]), np.array([180, 60,255]))
 
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
+def detect_friends_badge(img: np.ndarray,
+                         min_red: float = 0.003,
+                         min_white: float = 0.0015,
+                         corner_only: bool = True,
+                         margin: float = 0.24,
+                         pad: float = 0.12,
+                         min_white_overlap: float = 0.45):
+    """Detect TikTok "Friends" notification badges (bright red pill with white numerals)."""
+
+    def _mask_detector(region_mask: np.ndarray):
+        corner_px = max(int(np.count_nonzero(region_mask)), 1)
+        red_mask = cv2.bitwise_and(m_red, region_mask)
+        white_mask = cv2.bitwise_and(m_white, region_mask)
+        red_ratio = np.count_nonzero(red_mask) / corner_px
+        white_ratio = np.count_nonzero(white_mask) / corner_px
+        if red_ratio < min_red or white_ratio < min_white:
+            return None
+
+        white_pixels = np.count_nonzero(white_mask)
+        if white_pixels == 0:
+            return None
+
+        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
+        red_dilated = cv2.dilate(red_mask, dil_k, iterations=1)
+        overlap = np.count_nonzero(cv2.bitwise_and(red_dilated, white_mask)) / white_pixels
+        if overlap < min_white_overlap:
+            return None
+
+        combined = cv2.bitwise_or(red_mask, white_mask)
+        ys, xs = np.where(combined > 0)
+        if xs.size == 0 or ys.size == 0:
+            return None
+
+        x0, x1 = xs.min(), xs.max()
+        y0, y1 = ys.min(), ys.max()
+        w, h = x1 - x0 + 1, y1 - y0 + 1
+        dw, dh = int(w * pad), int(h * pad)
+        return (max(0, x0 - dw),
+                max(0, y0 - dh),
+                min(W - 1, x1 + dw),
+                min(H - 1, y1 + dh))
+
+    H, W = img.shape[:2]
+    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
+    m1 = cv2.inRange(hsv, RED_1[0], RED_1[1])
+    m2 = cv2.inRange(hsv, RED_2[0], RED_2[1])
+    m_red = cv2.bitwise_or(m1, m2)
+    m_white = cv2.inRange(hsv, WHITE[0], WHITE[1])
+
+    if corner_only:
+        for cm in corner_mask(img.shape, margin):
+            box = _mask_detector(cm)
+            if box is not None:
+                return True, box
+        return False, None
+
+    full_mask = np.full((H, W), 255, dtype=np.uint8)
+    box = _mask_detector(full_mask)
+    if box is not None:
+        return True, box
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
index 8480ba93dcb6135d45f7099f9253d11898394a65..eed961450ffbcc251b1faf674b653e8319e2f3d6 100644
--- a/tools/remove_tiktok_thumbs.py
+++ b/tools/remove_tiktok_thumbs.py
@@ -100,135 +164,153 @@ def detect_tiktok(img: np.ndarray, min_red=0.01, min_cyan=0.05,
                 x0, x1 = xs.min(), xs.max()
                 y0, y1 = ys.min(), ys.max()
                 # pad
                 w, h = x1 - x0 + 1, y1 - y0 + 1
                 dw, dh = int(w * pad), int(h * pad)
                 x0 = max(0, x0 - dw); y0 = max(0, y0 - dh)
                 x1 = min(W-1, x1 + dw); y1 = min(H-1, y1 + dh)
                 return True, (x0, y0, x1, y1)
         return False, None
 
     # global ratios (more conservative than aggressive)
     red_ratio = np.count_nonzero(m_red)  / (H * W)
     cyn_ratio = np.count_nonzero(m_cyan) / (H * W)
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
-                 min_red: float, min_cyan: float,
+                 min_red: float, min_cyan: float, min_white: float,
                  corner_only: bool, margin: float, pad: float,
                  aggressive: bool, rc_min: float):
     img = cv2.imread(str(fp))
     if img is None:
         return False, "unreadable"
     H, W = img.shape[:2]
 
     # Standard detector: corner → global fallback
     is_logo, box = detect_tiktok(img, min_red, min_cyan, corner_only, margin, pad=pad)
+    if not is_logo:
+        is_logo, box = detect_friends_badge(img, min_red=min_red, min_white=min_white,
+                                            corner_only=corner_only, margin=margin, pad=pad)
     if not is_logo and corner_only:
         is_logo, box = detect_tiktok(img, min_red, min_cyan, False, margin, pad=pad)
+        if not is_logo:
+            is_logo, box = detect_friends_badge(img, min_red=min_red, min_white=min_white,
+                                                corner_only=False, margin=margin, pad=pad)
 
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
         cv2.imwrite(str(fp), img)
         return True, "mask"
 
     return False, "noop"
 
 # --------------------------------- CLI -----------------------------------
 
 def main():
     p = argparse.ArgumentParser()
     p.add_argument("paths", nargs="+", help="folders to scan (e.g., public/thumbs ...)")
     p.add_argument("--mode", choices=["mask","black","delete","move"], default="black",
                    help="what to do with detected logos (default: black full-tile)")
     p.add_argument("--move-to", default=None, help="if --mode move, destination folder")
     p.add_argument("--min-red",  type=float, default=0.002,
                    help="red/pink ratio threshold (default tuned to 0.002)")
     p.add_argument("--min-cyan", type=float, default=0.002,
                    help="cyan ratio threshold (default tuned to 0.002)")
+    p.add_argument("--min-white", type=float, default=0.0015,
+                   help="white ratio threshold for Friends badge detection (default 0.0015)")
     p.add_argument("--corner-only", action="store_true",
                    help="only consider corners first (recommended)")
     p.add_argument("--margin", type=float, default=0.22,
                    help="corner window size as fraction of side length")
     p.add_argument("--pad", type=float, default=0.10,
                    help="padding around detected red/cyan region")
     p.add_argument("--aggressive", action="store_true",
                    help="add red–cyan overlap detector (global) as fallback")
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
+                        if not hit:
+                            hit, _ = detect_friends_badge(img, min_red=args.min_red,
+                                                          min_white=args.min_white,
+                                                          corner_only=args.corner_only,
+                                                          margin=args.margin, pad=args.pad)
                         if not hit and args.corner_only:
                             hit, _ = detect_tiktok(img, args.min_red, args.min_cyan,
                                                    False, args.margin, pad=args.pad)
+                            if not hit:
+                                hit, _ = detect_friends_badge(img, min_red=args.min_red,
+                                                              min_white=args.min_white,
+                                                              corner_only=False,
+                                                              margin=args.margin, pad=args.pad)
                         if not hit and args.aggressive:
                             hit = detect_tiktok_aggressive(img, min_red=args.min_red,
                                                            min_cyan=args.min_cyan, rc_min=args.rc_min)
                         if hit and args.move_to:
                             os.makedirs(args.move_to, exist_ok=True)
                             dest = Path(args.move_to) / name
                             if dest.exists():
                                 dest = dest.with_stem(dest.stem + "_dup")
                             fp.replace(dest)
                             actions.append({"file": str(fp), "action": "move", "dest": str(dest)})
                         continue
 
-                    changed, note = process_file(fp, args.mode, args.min_red, args.min_cyan,
+                    changed, note = process_file(fp, args.mode, args.min_red, args.min_cyan, args.min_white,
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
