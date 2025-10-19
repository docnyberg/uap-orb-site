import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
    import numpy as np
except Exception:
    print("Please install requirements:  pip install pillow numpy")
    sys.exit(1)


def read_list(csv_path):
    """Reads list from CSV/text."""
    csv_path = Path(csv_path)
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            rdr = csv.DictReader(f)
            fns = [r.get("file") or r.get("path") for r in rdr if (r.get("file") or r.get("path"))]
            if fns:
                return fns
    except Exception:
        pass
    with open(csv_path, encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_to_rel(p_raw, current_root: Path):
    p_raw = Path(p_raw)
    try:
        return str(p_raw.relative_to(current_root))
    except Exception:
        return p_raw.name


def load_img(path: Path):
    im = Image.open(path).convert("RGB")
    return im


def pil_to_tk(im: Image.Image, max_w=800, max_h=800):
    w, h = im.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return ImageTk.PhotoImage(im)


def newly_black_overlay(cur_im: Image.Image, bak_im: Image.Image, thr=12, tint=(255, 64, 64), alpha=128):
    cur = np.asarray(cur_im)
    bak = np.asarray(bak_im)
    if cur.shape != bak.shape:
        return cur_im

    cur_black = (cur <= thr).all(axis=2)
    bak_black = (bak <= thr).all(axis=2)
    mask = cur_black & (~bak_black)
    if not mask.any():
        return cur_im

    overlay = np.zeros_like(cur, dtype=np.uint8)
    overlay[mask] = np.array(tint, dtype=np.uint8)
    out = cur.copy()
    a = alpha / 255.0
    out[mask] = (a * overlay[mask] + (1 - a) * cur[mask]).astype(np.uint8)
    return Image.fromarray(out)


class Reviewer(tk.Tk):
    def __init__(self, files, current_root: Path, backup_root: Path, out_dir: Path,
                 black_thr=12, max_img_w=800, max_img_h=800):
        super().__init__()
        self.title("Side-by-Side Reviewer — Blacked vs Original")
        self.geometry("1720x980")
        self.files = files
        self.current_root = current_root
        self.backup_root = backup_root
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.black_thr = black_thr
        self.max_img_w = max_img_w
        self.max_img_h = max_img_h

        self.index = 0
        self.selected = set()
        self.cache = {}

        self._build_ui()
        self._bind_keys()
        self._render()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=8, pady=8)

        ctrl = ttk.Frame(top)
        ctrl.pack(fill="x", pady=(0, 8))

        self.lbl_status = ttk.Label(ctrl, text="0/0")
        self.lbl_status.pack(side="left", padx=(0, 12))

        self.var_overlay = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Show red overlay (newly black area)", variable=self.var_overlay, command=self._render).pack(side="left")

        ttk.Button(ctrl, text="Prev  ◀", command=self.prev_item).pack(side="left", padx=6)
        ttk.Button(ctrl, text="Next  ▶", command=self.next_item).pack(side="left", padx=6)

        self.btn_mark = ttk.Button(ctrl, text="Mark for Restore (Space)", command=self.toggle_mark)
        self.btn_mark.pack(side="left", padx=10)

        ttk.Button(ctrl, text="Save Selection", command=self.save_selection).pack(side="left", padx=10)
        ttk.Button(ctrl, text="Copy Selected", command=self.copy_selected).pack(side="left", padx=10)

        ttk.Button(ctrl, text="Open Current", command=self.open_current).pack(side="right", padx=6)
        ttk.Button(ctrl, text="Open Backup", command=self.open_backup).pack(side="right", padx=6)

        panels = ttk.Frame(top)
        panels.pack(fill="both", expand=True)

        left = ttk.Frame(panels)
        right = ttk.Frame(panels)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        right.pack(side="left", fill="both", expand=True, padx=(4, 0))

        self.lbl_left_title = ttk.Label(left, text="Current (blacked)", font=("Segoe UI", 10, "bold"))
        self.lbl_left_title.pack(anchor="w")
        self.canvas_left = tk.Label(left, bg="#111")
        self.canvas_left.pack(fill="both", expand=True)

        self.lbl_right_title = ttk.Label(right, text="Backup (original)", font=("Segoe UI", 10, "bold"))
        self.lbl_right_title.pack(anchor="w")
        self.canvas_right = tk.Label(right, bg="#111")
        self.canvas_right.pack(fill="both", expand=True)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.lbl_paths = ttk.Label(bottom, text="", justify="left")
        self.lbl_paths.pack(side="left", fill="x", expand=True)

        sel_box = ttk.Frame(self)
        sel_box.pack(fill="both", padx=8, pady=(0, 8))
        ttk.Label(sel_box, text="Selected for Restore:").pack(anchor="w")
        self.txt_selected = tk.Text(sel_box, height=6)
        self.txt_selected.pack(fill="both", expand=False)

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.prev_item())
        self.bind("<Right>", lambda e: self.next_item())
        self.bind("<space>", lambda e: self.toggle_mark())
        self.bind("<Return>", lambda e: self.next_item())
        self.bind("<Escape>", lambda e: self.quit())

    def cur_pair(self):
        rel = self.files[self.index]
        if re.match(r'^[A-Za-z]:\\', rel) or rel.startswith("\\\\"):
            rel = normalize_to_rel(rel, self.current_root)
        cur = self.current_root / rel
        bak = self.backup_root / rel
        return rel, cur, bak

    def _thumb_pair(self, rel, cur, bak):
        key = (rel, self.var_overlay.get())
        if key in self.cache:
            return self.cache[key]

        left_img = None
        right_img = None
        left_title = "Current (blacked)"
        right_title = "Backup (original)"

        try:
            cur_im = load_img(cur)
            bak_im = load_img(bak) if bak.exists() else None
            if bak_im and self.var_overlay.get():
                cur_im_disp = newly_black_overlay(cur_im, bak_im, thr=self.black_thr)
            else:
                cur_im_disp = cur_im
            left_img = pil_to_tk(cur_im_disp, self.max_img_w, self.max_img_h)
        except Exception:
            pass

        try:
            if bak.exists():
                right_img = pil_to_tk(load_img(bak), self.max_img_w, self.max_img_h)
            else:
                right_title += "  [MISSING]"
        except Exception:
            right_title += "  [ERROR OPENING]"

        self.cache[key] = (left_img, right_img, left_title, right_title)
        return self.cache[key]

    def _render(self):
        if not self.files:
            self.lbl_status.config(text="No files")
            return
        rel, cur, bak = self.cur_pair()
        status = f"{self.index + 1}/{len(self.files)}  —  {rel}"
        if rel in self.selected:
            status += "   [SELECTED]"
        self.lbl_status.config(text=status)

        left_img, right_img, left_title, right_title = self._thumb_pair(rel, cur, bak)
        self.lbl_left_title.config(text=left_title)
        self.lbl_right_title.config(text=right_title)

        self.canvas_left.config(image=left_img)
        self.canvas_left.image = left_img
        self.canvas_right.config(image=right_img)
        self.canvas_right.image = right_img

        self.lbl_paths.config(text=f"Current: {cur}\nBackup:  {bak}")
        self._refresh_selected_box()

        if rel in self.selected:
            self.btn_mark.config(text="Unmark (Space)")
        else:
            self.btn_mark.config(text="Mark for Restore (Space)")

    def _refresh_selected_box(self):
        self.txt_selected.delete("1.0", "end")
        for r in sorted(self.selected):
            self.txt_selected.insert("end", r + "\n")

    def prev_item(self):
        if not self.files:
            return
        self.index = (self.index - 1) % len(self.files)
        self._render()

    def next_item(self):
        if not self.files:
            return
        self.index = (self.index + 1) % len(self.files)
        self._render()

    def toggle_mark(self):
        if not self.files:
            return
        rel, cur, _ = self.cur_pair()
        if rel in self.selected:
            self.selected.remove(rel)
        else:
            self.selected.add(rel)
        self._render()

    def save_selection(self):
        restore_txt = self.out_dir / "restore_list.txt"
        review_csv = self.out_dir / "review.csv"
        with open(restore_txt, "w", encoding="utf-8") as f:
            for r in sorted(self.selected):
                f.write(r.replace("/", "\\") + "\n")
        with open(review_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["file", "selected_for_restore"])
            for r in self.files:
                orig_r = r
                if re.match(r'^[A-Za-z]:\\', r) or r.startswith("\\\\"):
                    orig_r = normalize_to_rel(r, self.current_root)
                wr.writerow([orig_r, 1 if orig_r in self.selected else 0])
        messagebox.showinfo("Saved", f"Saved:\n{restore_txt}\n{review_csv}")

    def copy_selected(self):
        if not self.selected:
            messagebox.showwarning("No selection", "No files marked for restore.")
            return
        if not messagebox.askyesno("Confirm", f"Copy {len(self.selected)} selected files from BACKUP to CURRENT?"):
            return
        errors = 0
        for rel in self.selected:
            src = self.backup_root / rel
            dst = self.current_root / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            except Exception:
                errors += 1
        if errors:
            messagebox.showwarning("Done (with errors)", f"Copied with {errors} errors.")
        else:
            messagebox.showinfo("Done", "Copied all selected files.")

    def open_current(self):
        _, cur, _ = self.cur_pair()
        try:
            os.startfile(cur)
        except Exception:
            pass

    def open_backup(self):
        _, _, bak = self.cur_pair()
        try:
            os.startfile(bak)
        except Exception:
            pass


def prompt_for_missing_args(args):
    need_prompt = not all([args.csv, args.current_root, args.backup_root, args.out_dir])
    if not need_prompt and not args.use_dialog:
        return args

    root = tk.Tk()
    root.withdraw()
    try:
        if args.use_dialog or not args.csv:
            csv_path = filedialog.askopenfilename(
                title="Select CSV or text list of blacked files",
                filetypes=[("CSV or text", "*.csv *.txt"), ("All files", "*.*")]
            )
            if not csv_path:
                sys.exit(0)
            args.csv = csv_path

        if args.use_dialog or not args.current_root:
            cur_dir = filedialog.askdirectory(title="Select CURRENT root folder", mustexist=True)
            if not cur_dir:
                sys.exit(0)
            args.current_root = cur_dir

        if args.use_dialog or not args.backup_root:
            bak_dir = filedialog.askdirectory(title="Select BACKUP root folder", mustexist=True)
            if not bak_dir:
                sys.exit(0)
            args.backup_root = bak_dir

        if args.use_dialog or not args.out_dir:
            out_dir = filedialog.askdirectory(title="Select or create OUTPUT folder", mustexist=False)
            if not out_dir:
                sys.exit(0)
            args.out_dir = out_dir
    finally:
        root.destroy()

    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV or text list of blacked files")
    ap.add_argument("--current-root")
    ap.add_argument("--backup-root")
    ap.add_argument("--out-dir")
    ap.add_argument("--use-dialog", action="store_true", help="Force GUI prompts for all paths")
    ap.add_argument("--black-thr", type=int, default=12, help="RGB threshold to consider 'black'")
    ap.add_argument("--max-img-w", type=int, default=820)
    ap.add_argument("--max-img-h", type=int, default=820)
    args = ap.parse_args()

    args = prompt_for_missing_args(args)

    if not all([args.csv, args.current_root, args.backup_root, args.out_dir]):
        print("Missing required paths.")
        sys.exit(1)

    current_root = Path(args.current_root)
    backup_root = Path(args.backup_root)
    out_dir = Path(args.out_dir)

    raw = read_list(args.csv)
    files = []
    for r in raw:
        r = r.strip()
        if not r:
            continue
        if re.match(r'^[A-Za-z]:\\', r) or r.startswith("\\\\"):
            r = normalize_to_rel(r, current_root)
        files.append(r)

    if not files:
        print("No files to review.")
        sys.exit(1)

    app = Reviewer(files, current_root, backup_root, out_dir,
                   black_thr=args.black_thr,
                   max_img_w=args.max_img_w,
                   max_img_h=args.max_img_h)
    app.mainloop()


if __name__ == "__main__":
    main()
