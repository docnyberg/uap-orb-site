import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk
    from PIL import Image, ImageTk
except Exception:
    print("Please install requirements:  pip install pillow")
    sys.exit(1)


THUMB_LIMIT = 120
THUMB_SIZE = 128
DEFAULT_COLUMNS = 6


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


@dataclass
class CuratorConfig:
    thumb_limit: int = THUMB_LIMIT
    thumb_size: int = THUMB_SIZE
    columns: int = DEFAULT_COLUMNS
    ui_mode: str = "navigator"


class HumanSymbolCurator(tk.Tk):
    def __init__(
        self,
        clusters: List[Cluster],
        members: Dict[int, List[str]],
        asset_base: Path,
        output_path: Path,
        config: Optional[CuratorConfig] = None,
    ):
        super().__init__()
        self.title("Human Symbol Curator")
        self.geometry("1580x960")

        self.clusters = clusters
        self.members = members
        self.asset_base = asset_base
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.config = config or CuratorConfig()
        self.thumb_size_var = tk.IntVar(value=max(32, self.config.thumb_size))
        self.thumb_limit_var = tk.IntVar(value=max(1, self.config.thumb_limit))
        self.columns_var = tk.IntVar(value=max(1, self.config.columns))

        self.cluster_order: List[int] = list(range(len(self.clusters)))
        self.current_pos = 0
        self.reviewed: Set[int] = set()

        self.renames: Dict[int, str] = {}
        self.flags: Dict[int, dict] = {}
        self.merges: Dict[int, dict] = {}
        self.excludes: Dict[int, dict] = {}

        self.thumbnail_cache: Dict[str, ImageTk.PhotoImage] = {}
        self.member_image_refs: List[ImageTk.PhotoImage] = []
        self.member_paths: List[str] = []

        self._filter_vars: Dict[str, tk.BooleanVar] = {}
        self._nav_selecting = False
        self.inspector_nav_holder: Optional[ttk.Frame] = None
        self.inspector_nav_visible: Optional[tk.BooleanVar] = None

        self._build_ui()
        self._bind_keys()
        self._apply_sort_filter()

        def _thumb_trace(*_: Sequence[object]):
            if self.cluster_order:
                self._render()

        self.thumb_size_var.trace_add("write", _thumb_trace)
        self.thumb_limit_var.trace_add("write", _thumb_trace)
        self.columns_var.trace_add("write", _thumb_trace)

    # UI ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=10, pady=10)

        header = ttk.Frame(top)
        header.pack(fill="x", pady=(0, 8))

        self.lbl_status = ttk.Label(header, text="0 / 0")
        self.lbl_status.pack(side="left")

        self.progress = ttk.Progressbar(header, length=160, mode="determinate")
        self.progress.pack(side="left", padx=(8, 16))

        ttk.Button(header, text="Prev", command=self.prev_cluster).pack(side="left", padx=(12, 4))
        ttk.Button(header, text="Next", command=self.next_cluster).pack(side="left", padx=4)

        ttk.Button(header, text="Rename", command=self.rename_cluster).pack(side="left", padx=(16, 4))
        ttk.Button(header, text="Merge With…", command=self.merge_with).pack(side="left", padx=4)
        ttk.Button(header, text="Flag", command=self.flag_cluster).pack(side="left", padx=4)
        ttk.Button(header, text="Exclude", command=self.exclude_cluster).pack(side="left", padx=4)
        ttk.Button(header, text="Save Actions", command=self.save_actions).pack(side="left", padx=(18, 4))

        jump_frame = ttk.Frame(header)
        jump_frame.pack(side="left", padx=(18, 4))
        ttk.Label(jump_frame, text="Jump to:").pack(side="left")
        self.entry_jump = ttk.Entry(jump_frame, width=8)
        self.entry_jump.pack(side="left", padx=4)
        ttk.Button(jump_frame, text="Go", command=self.jump_to_entry).pack(side="left")

        search_frame = ttk.Frame(header)
        search_frame.pack(side="left", padx=(16, 0))
        ttk.Label(search_frame, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        entry_search = ttk.Entry(search_frame, textvariable=self.search_var, width=18)
        entry_search.pack(side="left", padx=4)
        entry_search.bind("<Return>", lambda _: self.search_clusters())
        ttk.Button(search_frame, text="Find", command=self.search_clusters).pack(side="left")

        self.lbl_saved = ttk.Label(header, text="")
        self.lbl_saved.pack(side="right")

        if self.config.ui_mode == "navigator":
            content = ttk.Panedwindow(top, orient=tk.HORIZONTAL)
            content.pack(fill="both", expand=True)
            nav_holder = ttk.Frame(content, padding=(0, 0, 12, 0))
            content.add(nav_holder, weight=1)
            detail_holder = ttk.Frame(content)
            content.add(detail_holder, weight=4)
            self._build_nav_panel(nav_holder)
            self._build_detail_panel(detail_holder)
        else:
            content = ttk.Frame(top)
            content.pack(fill="both", expand=True)

            body = ttk.Frame(content)
            body.pack(fill="both", expand=True)
            body.columnconfigure(1, weight=1)
            body.rowconfigure(0, weight=1)

            self.inspector_nav_holder = ttk.Frame(body, padding=(0, 0, 12, 0))
            self._build_nav_panel(self.inspector_nav_holder)

            detail_holder = ttk.Frame(body)
            detail_holder.grid(row=0, column=1, sticky="nsew")
            self._build_detail_panel(detail_holder)

            self.inspector_nav_visible = tk.BooleanVar(value=False)
            self._build_inspector_toolbar(content)
            self._toggle_inspector_nav(False)

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.prev_cluster())
        self.bind("<Right>", lambda e: self.next_cluster())
        self.bind("<Escape>", lambda e: self.quit())
        self.bind("<Control-s>", lambda e: self.save_actions())
        self.bind("<Control-r>", lambda e: self.rename_cluster())
        self.bind("<Control-m>", lambda e: self.merge_with())
        self.bind("<Control-f>", lambda e: self.flag_cluster())
        self.bind("<Control-e>", lambda e: self.exclude_cluster())
        self.bind("<Control-j>", lambda e: self.prompt_jump())

    def _build_nav_panel(self, parent: ttk.Frame):
        search_box = ttk.Labelframe(parent, text="Navigation")
        search_box.pack(fill="x", pady=(0, 12))

        ttk.Label(search_box, text="Sort by:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.sort_var = tk.StringVar(value="ID")
        sort_combo = ttk.Combobox(
            search_box,
            state="readonly",
            textvariable=self.sort_var,
            values=["ID", "Name", "Count", "Pending"],
        )
        sort_combo.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        sort_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_sort_filter())

        filter_box = ttk.Frame(search_box)
        filter_box.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2)
        for idx, (key, label) in enumerate(
            [
                ("renamed", "Renamed"),
                ("merged", "Merged"),
                ("flagged", "Flagged"),
                ("excluded", "Excluded"),
            ]
        ):
            var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(filter_box, text=label, variable=var, command=self._apply_sort_filter)
            chk.grid(row=0, column=idx, padx=2, pady=2, sticky="w")
            self._filter_vars[key] = var

        ttk.Label(search_box, text="Thumb limit:").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        ttk.Spinbox(search_box, from_=10, to=1000, increment=10, textvariable=self.thumb_limit_var, width=8).grid(
            row=2, column=1, padx=4, pady=4, sticky="w"
        )

        ttk.Label(search_box, text="Columns:").grid(row=3, column=0, padx=4, pady=4, sticky="w")
        ttk.Spinbox(search_box, from_=1, to=12, textvariable=self.columns_var, width=8).grid(
            row=3, column=1, padx=4, pady=4, sticky="w"
        )

        ttk.Label(search_box, text="Thumb size:").grid(row=4, column=0, padx=4, pady=4, sticky="w")
        ttk.Spinbox(search_box, from_=32, to=256, increment=16, textvariable=self.thumb_size_var, width=8).grid(
            row=4, column=1, padx=4, pady=4, sticky="w"
        )

        search_box.columnconfigure(1, weight=1)

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True)
        self.nav_listbox = tk.Listbox(list_frame)
        self.nav_listbox.pack(fill="both", expand=True)
        self.nav_listbox.bind("<<ListboxSelect>>", self._on_nav_select)

    def _build_detail_panel(self, parent: ttk.Frame):
        content = ttk.Frame(parent)
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content)
        left.pack(side="left", fill="y", padx=(0, 12))

        ttk.Label(left, text="Prototype", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.lbl_cluster = ttk.Label(left, text="", justify="left")
        self.lbl_cluster.pack(anchor="w", pady=(4, 12))

        self.lbl_proto = tk.Label(left, bg="#111")
        self.lbl_proto.pack(fill="both", expand=True)

        actions_box = ttk.Labelframe(left, text="Recorded Actions")
        actions_box.pack(fill="both", pady=(12, 0), expand=True)
        self.txt_actions = tk.Text(actions_box, height=8, width=40)
        self.txt_actions.pack(fill="both", expand=True)

        right = ttk.Frame(content)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Cluster Members", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True, pady=(4, 0))

        grid_tab = ttk.Frame(notebook)
        notebook.add(grid_tab, text="Grid")

        canvas = tk.Canvas(grid_tab, bg="#111")
        canvas.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(grid_tab, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.members_frame = ttk.Frame(canvas)
        self.members_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.members_frame, anchor="nw")
        self.canvas_members = canvas

        list_tab = ttk.Frame(notebook)
        self.members_tree = ttk.Treeview(list_tab, columns=("member",), show="headings")
        self.members_tree.heading("member", text="Member Thumb Path")
        self.members_tree.pack(fill="both", expand=True)
        self.members_tree.bind("<Double-1>", self._on_tree_double_click)

        if self.config.ui_mode == "inspector":
            notebook.add(list_tab, text="List")

        self.members_notebook = notebook

    def _build_inspector_toolbar(self, parent: ttk.Frame):
        toolbar = ttk.Labelframe(parent, text="Inspector Controls")
        toolbar.pack(fill="x", pady=(8, 0))

        ttk.Label(toolbar, text="Thumb limit:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Spinbox(toolbar, from_=10, to=2000, increment=10, textvariable=self.thumb_limit_var, width=8).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )

        ttk.Label(toolbar, text="Columns:").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Spinbox(toolbar, from_=1, to=12, textvariable=self.columns_var, width=6).grid(
            row=0, column=3, padx=4, pady=4, sticky="w"
        )

        ttk.Label(toolbar, text="Thumb size:").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        ttk.Scale(toolbar, from_=48, to=256, variable=self.thumb_size_var).grid(
            row=0, column=5, padx=4, pady=4, sticky="ew"
        )

        toolbar.columnconfigure(5, weight=1)

        # Quick access to navigator perks without opening the drawer.
        if hasattr(self, "sort_var"):
            ttk.Label(toolbar, text="Sort:").grid(row=1, column=0, padx=4, pady=(0, 4), sticky="w")
            sort_combo = ttk.Combobox(
                toolbar,
                state="readonly",
                textvariable=self.sort_var,
                values=["ID", "Name", "Count", "Pending"],
                width=10,
            )
            sort_combo.grid(row=1, column=1, padx=4, pady=(0, 4), sticky="w")
            sort_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_sort_filter())

        col = 2
        for key, label in [
            ("renamed", "Renamed"),
            ("merged", "Merged"),
            ("flagged", "Flagged"),
            ("excluded", "Excluded"),
        ]:
            var = self._filter_vars.get(key)
            if not var:
                continue
            ttk.Checkbutton(toolbar, text=label, variable=var, command=self._apply_sort_filter).grid(
                row=1, column=col, padx=2, pady=(0, 4), sticky="w"
            )
            col += 1

        if self.inspector_nav_visible is not None:
            ttk.Checkbutton(
                toolbar,
                text="Navigator drawer",
                variable=self.inspector_nav_visible,
                command=self._toggle_inspector_nav,
            ).grid(row=1, column=col, padx=6, pady=(0, 4), sticky="e")

    # Navigation ----------------------------------------------------------
    def prev_cluster(self):
        if self.current_pos > 0:
            self.current_pos -= 1
            self._render()

    def next_cluster(self):
        if self.cluster_order and self.current_pos < len(self.cluster_order) - 1:
            self.current_pos += 1
            self._render()

    def jump_to_entry(self):
        target = self.entry_jump.get().strip()
        if target:
            self._jump_to_value(target)

    def prompt_jump(self):
        value = simpledialog.askstring("Jump", "Enter cluster ID or name:", parent=self)
        if value:
            self._jump_to_value(value.strip())

    def _jump_to_value(self, value: str):
        if not self.cluster_order:
            return
        try:
            cid = int(value)
        except ValueError:
            cid = None
        if cid is not None:
            for pos, idx in enumerate(self.cluster_order):
                if self.clusters[idx].cluster_id == cid:
                    self.current_pos = pos
                    self._render()
                    return
        lower = value.lower()
        for pos, idx in enumerate(self.cluster_order):
            cluster = self.clusters[idx]
            name = self.renames.get(cluster.cluster_id) or cluster.name
            if lower in name.lower():
                self.current_pos = pos
                self._render()
                return
        messagebox.showinfo("Jump", f"No cluster found for '{value}'.")

    def search_clusters(self):
        if not self.cluster_order:
            return
        query = self.search_var.get().strip().lower()
        if not query:
            return
        start_pos = (self.current_pos + 1) % len(self.cluster_order)
        for offset in range(len(self.cluster_order)):
            pos = (start_pos + offset) % len(self.cluster_order)
            idx = self.cluster_order[pos]
            cluster = self.clusters[idx]
            name = self.renames.get(cluster.cluster_id) or cluster.name
            if query in name.lower():
                self.current_pos = pos
                self._render()
                return
        messagebox.showinfo("Search", f"No additional clusters match '{query}'.")

    # Rendering -----------------------------------------------------------
    def _render(self):
        if not self.cluster_order:
            self.lbl_status.config(text="No clusters loaded")
            self.lbl_proto.config(image="")
            self.lbl_proto.image = None
            self._render_members(-1)
            self._render_actions(-1)
            return
        idx = self.cluster_order[self.current_pos]
        cluster = self.clusters[idx]
        self.reviewed.add(cluster.cluster_id)

        total = len(self.cluster_order)
        self.lbl_status.config(text=f"{self.current_pos + 1} / {total}  (ID {cluster.cluster_id})")
        reviewed_in_scope = sum(
            1 for idx2 in self.cluster_order if self.clusters[idx2].cluster_id in self.reviewed
        )
        self.progress.config(maximum=max(total, 1), value=reviewed_in_scope)

        if hasattr(self, "nav_listbox"):
            self._nav_selecting = True
            try:
                self.nav_listbox.selection_clear(0, tk.END)
                self.nav_listbox.selection_set(self.current_pos)
                self.nav_listbox.see(self.current_pos)
            finally:
                self._nav_selecting = False

        rename = self.renames.get(cluster.cluster_id)
        name_display = rename or cluster.name
        info = (
            f"ID: {cluster.cluster_id}\n"
            f"Name: {name_display}\n"
            f"Count: {cluster.count}\n"
            f"Badges: {self._cluster_badge_text(cluster.cluster_id) or '—'}"
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
        self.member_paths.clear()
        if hasattr(self, "members_tree"):
            for item in self.members_tree.get_children():
                self.members_tree.delete(item)

        if cluster_id == -1:
            return

        limit = max(1, int(self.thumb_limit_var.get()))
        thumbs = self.members.get(cluster_id, [])[:limit]
        if not thumbs:
            ttk.Label(self.members_frame, text="No members found.").grid(row=0, column=0, padx=6, pady=6, sticky="w")
            return

        columns = max(1, int(self.columns_var.get()))
        thumb_size = max(32, int(self.thumb_size_var.get()))
        for idx, thumb_rel in enumerate(thumbs):
            path = resolve_asset(self.asset_base, thumb_rel)
            thumb_img = None
            if path:
                cache_key = f"{path}|{thumb_size}"
                thumb_img = self.thumbnail_cache.get(cache_key)
                if thumb_img is None:
                    thumb_img = load_image_thumb(path, size=thumb_size)
                    if thumb_img:
                        self.thumbnail_cache[cache_key] = thumb_img
            frame = ttk.Frame(self.members_frame, padding=4)
            frame.grid(row=idx // columns, column=idx % columns, sticky="n")
            lbl = tk.Label(frame, image="" if not thumb_img else thumb_img, bg="#111")
            lbl.pack()
            if path:
                lbl.bind("<Button-1>", lambda _e, p=path: self.open_member_viewer(p))
            if thumb_img:
                self.member_image_refs.append(thumb_img)
            ttk.Label(frame, text=thumb_rel, wraplength=thumb_size * 1.2).pack()
            self.member_paths.append(thumb_rel)
            if hasattr(self, "members_tree"):
                self.members_tree.insert("", tk.END, values=(thumb_rel,))

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
        if cluster_id in self.excludes:
            actions.append({"type": "exclude", **self.excludes[cluster_id]})
        self.txt_actions.delete("1.0", tk.END)
        if not actions:
            self.txt_actions.insert(tk.END, "No actions recorded.")
        else:
            self.txt_actions.insert(tk.END, json.dumps(actions, indent=2))

    def _cluster_badge_text(self, cluster_id: int) -> str:
        badges = []
        if cluster_id in self.renames:
            badges.append("R")
        if cluster_id in self.merges:
            badges.append("M")
        if cluster_id in self.flags:
            badges.append("F")
        if cluster_id in self.excludes:
            badges.append("X")
        return "".join(badges)

    def _cluster_matches_filters(self, idx: int) -> bool:
        if not self._filter_vars:
            return True
        active = [key for key, var in self._filter_vars.items() if var.get()]
        if not active:
            return True
        cluster_id = self.clusters[idx].cluster_id
        mapping = {
            "renamed": cluster_id in self.renames,
            "merged": cluster_id in self.merges,
            "flagged": cluster_id in self.flags,
            "excluded": cluster_id in self.excludes,
        }
        return any(mapping.get(key, False) for key in active)

    def _cluster_pending_weight(self, idx: int) -> int:
        cluster_id = self.clusters[idx].cluster_id
        has_action = any(
            mapping
            for mapping in (
                cluster_id in self.renames,
                cluster_id in self.merges,
                cluster_id in self.flags,
                cluster_id in self.excludes,
            )
        )
        return 0 if has_action else 1

    def _apply_sort_filter(self):
        previous_cluster_id = None
        if self.cluster_order:
            previous_cluster_id = self.clusters[self.cluster_order[self.current_pos]].cluster_id
        indices = [idx for idx in range(len(self.clusters)) if self._cluster_matches_filters(idx)]
        sort_key = getattr(self, "sort_var", None)
        sort_value = sort_key.get() if sort_key else "ID"
        if sort_value == "Name":
            indices.sort(key=lambda idx: (self.renames.get(self.clusters[idx].cluster_id) or self.clusters[idx].name).lower())
        elif sort_value == "Count":
            indices.sort(key=lambda idx: self.clusters[idx].count, reverse=True)
        elif sort_value == "Pending":
            indices.sort(key=lambda idx: (self._cluster_pending_weight(idx), self.clusters[idx].cluster_id))
        else:
            indices.sort(key=lambda idx: self.clusters[idx].cluster_id)
        self.cluster_order = indices
        if not self.cluster_order:
            self.current_pos = 0
        else:
            if previous_cluster_id is not None:
                for pos, idx in enumerate(self.cluster_order):
                    if self.clusters[idx].cluster_id == previous_cluster_id:
                        self.current_pos = pos
                        break
                else:
                    self.current_pos = 0
            else:
                self.current_pos = 0
        if hasattr(self, "nav_listbox"):
            self._refresh_nav_list()
        self._render()

    def _refresh_nav_list(self):
        if not hasattr(self, "nav_listbox"):
            return
        self.nav_listbox.delete(0, tk.END)
        for idx in self.cluster_order:
            cluster = self.clusters[idx]
            badges = self._cluster_badge_text(cluster.cluster_id)
            badge_text = f" [{badges}]" if badges else ""
            label = f"{cluster.cluster_id}: {cluster.name} ({cluster.count}){badge_text}"
            self.nav_listbox.insert(tk.END, label)

    def _on_nav_select(self, _event):
        if self._nav_selecting:
            return
        selection = self.nav_listbox.curselection()
        if not selection:
            return
        self.current_pos = selection[0]
        self._render()

    def _toggle_inspector_nav(self, force: Optional[bool] = None):
        if not self.inspector_nav_holder or not self.inspector_nav_visible:
            return
        if force is not None:
            self.inspector_nav_visible.set(bool(force))
        show = bool(self.inspector_nav_visible.get())
        if show:
            self.inspector_nav_holder.grid(row=0, column=0, sticky="ns")
            self._refresh_nav_list()
        else:
            if self.inspector_nav_holder.winfo_manager() == "grid":
                self.inspector_nav_holder.grid_remove()

    def _on_tree_double_click(self, _event):
        item = self.members_tree.selection()
        if not item:
            return
        thumb_rel = self.members_tree.item(item[0], "values")[0]
        path = resolve_asset(self.asset_base, thumb_rel)
        if path:
            self.open_member_viewer(path)

    def open_member_viewer(self, path: Path):
        if not path or not path.exists():
            messagebox.showinfo("View", "Image not available.")
            return
        MemberViewer(self, path)

    # Actions -------------------------------------------------------------
    def rename_cluster(self):
        if not self.cluster_order:
            return
        cluster = self.clusters[self.cluster_order[self.current_pos]]
        current_name = self.renames.get(cluster.cluster_id) or cluster.name
        new_name = simpledialog.askstring(
            "Rename Cluster",
            "Enter new cluster name:",
            initialvalue=current_name,
            parent=self,
        )
        if new_name:
            self.renames[cluster.cluster_id] = new_name.strip()
            self._apply_sort_filter()

    def merge_with(self):
        if not self.cluster_order:
            return
        cluster = self.clusters[self.cluster_order[self.current_pos]]
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
        self._apply_sort_filter()

    def flag_cluster(self):
        if not self.cluster_order:
            return
        cluster = self.clusters[self.cluster_order[self.current_pos]]
        reason = simpledialog.askstring("Flag for Review", "Describe the issue:", parent=self)
        if reason:
            self.flags[cluster.cluster_id] = {"flag": "needs_attention", "reason": reason.strip()}
            self._apply_sort_filter()

    def exclude_cluster(self):
        if not self.cluster_order:
            return
        cluster = self.clusters[self.cluster_order[self.current_pos]]
        existing = self.excludes.get(cluster.cluster_id)
        prompt = (
            "Provide reason for exclusion:"
            if not existing
            else "Update exclusion reason (leave blank to remove):"
        )
        reason = simpledialog.askstring(
            "Exclude Cluster",
            prompt,
            initialvalue=existing.get("reason") if existing else "",
            parent=self,
        )
        if reason is None:
            return
        reason = reason.strip()
        if not reason:
            self.excludes.pop(cluster.cluster_id, None)
        else:
            self.excludes[cluster.cluster_id] = {"reason": reason}
        self._apply_sort_filter()

    def save_actions(self):
        data = {
            "merges": [],
            "renames": [],
            "flags": [],
            "excludes": [],
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
        for cid, info in sorted(self.excludes.items()):
            payload = {"cluster_id": cid}
            payload.update(info)
            data["excludes"].append(payload)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.lbl_saved.config(text=f"Saved to {self.output_path}")
        self._refresh_nav_list()


class MemberViewer(tk.Toplevel):
    def __init__(self, parent: "HumanSymbolCurator", path: Path):
        super().__init__(parent)
        self.title(path.name)
        self.geometry("540x600")
        self.transient(parent)

        self._path = path
        self._original = Image.open(path).convert("RGB")
        self._photo: Optional[ImageTk.PhotoImage] = None

        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#111")
        self.canvas.pack(fill="both", expand=True)

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text=str(path)).pack(side="left", padx=6)
        ttk.Label(controls, text="Zoom:").pack(side="left", padx=6)
        self.zoom_var = tk.DoubleVar(value=1.0)
        slider = ttk.Scale(controls, from_=0.2, to=3.0, variable=self.zoom_var, command=lambda _evt: self._update_image())
        slider.pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(controls, text="Reset", command=self._reset_zoom).pack(side="right", padx=6)

        self.bind("<Escape>", lambda _e: self.destroy())
        self._update_image()

    def _reset_zoom(self):
        self.zoom_var.set(1.0)
        self._update_image()

    def _update_image(self):
        zoom = max(0.2, float(self.zoom_var.get()))
        width, height = self._original.size
        new_size = (max(1, int(width * zoom)), max(1, int(height * zoom)))
        resized = self._original.resize(new_size, Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(10, 10, anchor="nw", image=self._photo)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))


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
    parser.add_argument("--thumb-limit", type=int, default=THUMB_LIMIT, help="Maximum member thumbnails to display per cluster.")
    parser.add_argument("--thumb-size", type=int, default=THUMB_SIZE, help="Thumbnail size (pixels) for member thumbnails.")
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS, help="Number of thumbnail columns in the grid view.")
    parser.add_argument(
        "--ui-mode",
        choices=("navigator", "inspector"),
        default="navigator",
        help="Navigator adds jump/search list management, inspector emphasizes comparison tooling.",
    )
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

    config = CuratorConfig(
        thumb_limit=max(1, args.thumb_limit),
        thumb_size=max(32, args.thumb_size),
        columns=max(1, args.columns),
        ui_mode=args.ui_mode,
    )

    app = HumanSymbolCurator(clusters, members, asset_base, args.output, config)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
