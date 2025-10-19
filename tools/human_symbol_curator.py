import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk
    from PIL import Image, ImageTk
except Exception:
    print("Please install requirements:  pip install pillow")
    sys.exit(1)


THUMB_LIMIT = 120
THUMB_SIZE = 128


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_asset(base: Path, rel_path: str) -> Optional[Path]:
    if not rel_path:
        return None
    rel = Path(rel_path)
    if rel.is_absolute():
        return rel if rel.exists() else None
    cand = base / rel
    if cand.exists():
        return cand
    return None


def load_image_thumb(path: Path, size: int = THUMB_SIZE) -> Optional[ImageTk.PhotoImage]:
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        scale = min(size / max(w, 1), size / max(h, 1), 1.0)
        if scale != 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return ImageTk.PhotoImage(im)
    except Exception:
        return None


class Cluster:
    def __init__(self, data: dict):
        self.cluster_id: int = data.get("cluster_id")
        self.name: str = (
            data.get("name")
            or data.get("token")
            or data.get("canonical_name")
            or f"Cluster {self.cluster_id}"
        )
        proto = data.get("prototype", "")
        if isinstance(proto, dict):
            proto = proto.get("thumb_obj", "") or proto.get("thumb", "")
        self.prototype: str = proto
        self.name: str = data.get("name") or f"Cluster {self.cluster_id}"
        self.prototype: str = data.get("prototype", "")
        self.count: int = data.get("count", 0)


def build_member_index(seq_paths: List[Path]) -> Dict[int, List[str]]:
    members: Dict[int, List[str]] = {}
    for seq_path in seq_paths:
        try:
            seq = load_json(seq_path)
        except Exception:
            continue
        for event in seq.get("events", []):
            cid = event.get("cluster_id")
            thumb = event.get("thumb_obj")
            if cid is None or not thumb:
                continue
            members.setdefault(cid, []).append(thumb)
    return members


class HumanSymbolCurator(tk.Tk):
    def __init__(self, clusters: List[Cluster], members: Dict[int, List[str]], asset_base: Path, output_path: Path):
        super().__init__()
        self.title("Human Symbol Curator")
        self.geometry("1580x960")

        self.clusters = clusters
        self.members = members
        self.asset_base = asset_base
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.index = 0

        self.renames: Dict[int, str] = {}
        self.flags: Dict[int, dict] = {}
        self.merges: Dict[int, dict] = {}

        self.thumbnail_cache: Dict[str, ImageTk.PhotoImage] = {}
        self.member_image_refs: List[ImageTk.PhotoImage] = []

        self._build_ui()
        self._bind_keys()
        self._render()

    # UI ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=10, pady=10)

        header = ttk.Frame(top)
        header.pack(fill="x", pady=(0, 8))

        self.lbl_status = ttk.Label(header, text="0 / 0")
        self.lbl_status.pack(side="left")

        ttk.Button(header, text="Prev", command=self.prev_cluster).pack(side="left", padx=(12, 4))
        ttk.Button(header, text="Next", command=self.next_cluster).pack(side="left", padx=4)

        ttk.Button(header, text="Rename", command=self.rename_cluster).pack(side="left", padx=(16, 4))
        ttk.Button(header, text="Merge With…", command=self.merge_with).pack(side="left", padx=4)
        ttk.Button(header, text="Flag for Review", command=self.flag_cluster).pack(side="left", padx=4)
        ttk.Button(header, text="Save Actions", command=self.save_actions).pack(side="left", padx=(18, 4))

        self.lbl_saved = ttk.Label(header, text="")
        self.lbl_saved.pack(side="right")

        content = ttk.Frame(top)
        content.pack(fill="both", expand=True)

        # Prototype panel -------------------------------------------------
        left = ttk.Frame(content)
        left.pack(side="left", fill="y", padx=(0, 12))

        ttk.Label(left, text="Prototype", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.lbl_cluster = ttk.Label(left, text="", justify="left")
        self.lbl_cluster.pack(anchor="w", pady=(4, 12))

        self.lbl_proto = tk.Label(left, bg="#111")
        self.lbl_proto.pack(fill="both", expand=True)

        actions_box = ttk.Labelframe(left, text="Recorded Actions")
        actions_box.pack(fill="x", pady=(12, 0))
        self.txt_actions = tk.Text(actions_box, height=8, width=40)
        self.txt_actions.pack(fill="both", expand=True)

        # Members grid ----------------------------------------------------
        right = ttk.Frame(content)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Cluster Members", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        canvas = tk.Canvas(right, bg="#111")
        canvas.pack(fill="both", expand=True, pady=(4, 0))
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.members_frame = ttk.Frame(canvas)
        self.members_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.members_frame, anchor="nw")
        self.canvas_members = canvas

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.prev_cluster())
        self.bind("<Right>", lambda e: self.next_cluster())
        self.bind("<Escape>", lambda e: self.quit())
        self.bind("<Control-s>", lambda e: self.save_actions())
        self.bind("<Control-r>", lambda e: self.rename_cluster())

    # Navigation ----------------------------------------------------------
    def prev_cluster(self):
        if self.index > 0:
            self.index -= 1
            self._render()

    def next_cluster(self):
        if self.index < len(self.clusters) - 1:
            self.index += 1
            self._render()

    # Rendering -----------------------------------------------------------
    def _render(self):
        if not self.clusters:
            self.lbl_status.config(text="No clusters loaded")
            return
        cluster = self.clusters[self.index]
        self.lbl_status.config(text=f"{self.index + 1} / {len(self.clusters)}")

        rename = self.renames.get(cluster.cluster_id)
        name_display = rename or cluster.name
        info = (
            f"ID: {cluster.cluster_id}\n"
            f"Name: {name_display}\n"
            f"Count: {cluster.count}"
        )
        self.lbl_cluster.config(text=info)

        proto_path = resolve_asset(self.asset_base, cluster.prototype)
        proto_img = None
        if proto_path:
            proto_img = load_image_thumb(proto_path, size=320)
        self.lbl_proto.config(image="" if not proto_img else proto_img)
        self.lbl_proto.image = proto_img

        self._render_members(cluster.cluster_id)
        self._render_actions(cluster.cluster_id)
        self.lbl_saved.config(text="")

    def _render_members(self, cluster_id: int):
        for child in self.members_frame.winfo_children():
            child.destroy()
        self.member_image_refs.clear()

        thumbs = self.members.get(cluster_id, [])[:THUMB_LIMIT]
        if not thumbs:
            ttk.Label(self.members_frame, text="No members found.").grid(row=0, column=0, padx=6, pady=6, sticky="w")
            return

        columns = 6
        for idx, thumb_rel in enumerate(thumbs):
            path = resolve_asset(self.asset_base, thumb_rel)
            thumb_img = None
            if path:
                cache_key = str(path)
                thumb_img = self.thumbnail_cache.get(cache_key)
                if thumb_img is None:
                    thumb_img = load_image_thumb(path)
                    if thumb_img:
                        self.thumbnail_cache[cache_key] = thumb_img
            frame = ttk.Frame(self.members_frame, padding=4)
            frame.grid(row=idx // columns, column=idx % columns, sticky="n")
            lbl = tk.Label(frame, image="" if not thumb_img else thumb_img, bg="#111")
            lbl.pack()
            if thumb_img:
                self.member_image_refs.append(thumb_img)
            ttk.Label(frame, text=thumb_rel, wraplength=THUMB_SIZE * 1.2).pack()

    def _render_actions(self, cluster_id: int):
        actions = []
        if cluster_id in self.renames:
            actions.append({"type": "rename", "new_name": self.renames[cluster_id]})
        merge = self.merges.get(cluster_id)
        if merge:
            act = {"type": "merge", "sources": merge.get("source_ids", [])}
            if merge.get("new_name"):
                act["new_name"] = merge["new_name"]
            actions.append(act)
        if cluster_id in self.flags:
            actions.append({"type": "flag", **self.flags[cluster_id]})
        self.txt_actions.delete("1.0", tk.END)
        if not actions:
            self.txt_actions.insert(tk.END, "No actions recorded.")
        else:
            self.txt_actions.insert(tk.END, json.dumps(actions, indent=2))

    # Actions -------------------------------------------------------------
    def rename_cluster(self):
        cluster = self.clusters[self.index]
        current_name = self.renames.get(cluster.cluster_id) or cluster.name
        new_name = simpledialog.askstring("Rename Cluster", "Enter new cluster name:", initialvalue=current_name, parent=self)
        if new_name:
            self.renames[cluster.cluster_id] = new_name.strip()
            self._render()

    def merge_with(self):
        cluster = self.clusters[self.index]
        dialog = MergeDialog(
            self,
            self.clusters,
            current_id=cluster.cluster_id,
            existing=self.merges.get(cluster.cluster_id),
        )
        self.wait_window(dialog)
        if not dialog.result:
            return
        self.merges[cluster.cluster_id] = dialog.result
        self._render()

    def flag_cluster(self):
        cluster = self.clusters[self.index]
        reason = simpledialog.askstring("Flag for Review", "Describe the issue:", parent=self)
        if reason:
            self.flags[cluster.cluster_id] = {"flag": "needs_attention", "reason": reason.strip()}
            self._render()

    def save_actions(self):
        data = {
            "merges": [],
            "renames": [],
            "flags": [],
        }
        for target_id, merge in sorted(self.merges.items()):
            record = {
                "target_id": target_id,
                "source_ids": merge.get("source_ids", []),
            }
            if merge.get("new_name"):
                record["new_name"] = merge["new_name"]
            data["merges"].append(record)
        for cid, name in sorted(self.renames.items()):
            data["renames"].append({"cluster_id": cid, "new_name": name})
        for cid, info in sorted(self.flags.items()):
            payload = {"cluster_id": cid}
            payload.update(info)
            data["flags"].append(payload)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.lbl_saved.config(text=f"Saved to {self.output_path}")


class MergeDialog(tk.Toplevel):
    def __init__(
        self,
        parent: HumanSymbolCurator,
        clusters: List[Cluster],
        current_id: int,
        existing: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.title("Merge Clusters")
        self.geometry("420x520")
        self.transient(parent)
        self.grab_set()

        ttk.Label(self, text=f"Merge into cluster {current_id}").pack(pady=(8, 4))
        ttk.Label(self, text="Select clusters to merge:").pack()

        self.listbox = tk.Listbox(self, selectmode=tk.MULTIPLE)
        self.listbox.pack(fill="both", expand=True, padx=10, pady=6)

        self.cluster_options: List[int] = []
        for cluster in clusters:
            if cluster.cluster_id == current_id:
                continue
            label = f"{cluster.cluster_id} — {cluster.name}"
            self.cluster_options.append(cluster.cluster_id)
            self.listbox.insert(tk.END, label)

        if existing:
            existing_sources = set(existing.get("source_ids", []))
            for idx, cid in enumerate(self.cluster_options):
                if cid in existing_sources:
                    self.listbox.selection_set(idx)

        ttk.Label(self, text="Optional new name:").pack(pady=(6, 0))
        self.entry_name = ttk.Entry(self)
        self.entry_name.pack(fill="x", padx=10)
        if existing and existing.get("new_name"):
            self.entry_name.insert(0, existing["new_name"])

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(side="left", padx=6)

        self.result = None

    def _apply(self):
        selections = self.listbox.curselection()
        if not selections:
            messagebox.showinfo("Merge", "Select at least one cluster to merge.")
            return
        source_ids = []
        for idx in selections:
            try:
                cid = self.cluster_options[idx]
            except IndexError:
                continue
            source_ids.append(cid)
        result = {"source_ids": source_ids}
        new_name = self.entry_name.get().strip()
        if new_name:
            result["new_name"] = new_name
        self.result = result
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Curate symbol clusters with human-in-the-loop review.")
    parser.add_argument("codebook", type=Path, help="Path to symbol_codebook_canon.json")
    parser.add_argument("sequences", type=Path, help="Path to directory containing *.sequence.json files")
    parser.add_argument("thumb_root", type=Path, help="Root folder that contains thumbs_obj/")
    parser.add_argument("output", type=Path, help="Where to write human_review_actions.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    codebook_data = load_json(args.codebook)
    clusters = [Cluster(c) for c in codebook_data.get("clusters", [])]
    if not clusters:
        print("No clusters found in codebook.")
        return 1

    seq_paths = sorted(args.sequences.glob("*.sequence.json"))
    if not seq_paths:
        print("No sequence files found.")
        return 1

    members = build_member_index(seq_paths)

    asset_base = args.thumb_root
    if not asset_base.exists():
        print(f"Asset base not found: {asset_base}")
        return 1

    app = HumanSymbolCurator(clusters, members, asset_base, args.output)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
