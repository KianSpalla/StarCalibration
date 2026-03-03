import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from stellarcalibrationbackend import run_calibration

BG       = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
ACCENT_H = "#79b8ff"
FG       = "#e6edf3"
FG_DIM   = "#8b949e"
SUCCESS  = "#3fb950"
ERROR    = "#f85149"
WARN     = "#d29922"

FONT       = ("Segoe UI", 10)
FONT_SB    = ("Segoe UI", 10, "bold")
FONT_LG    = ("Segoe UI", 12, "bold")
FONT_TITLE = ("Segoe UI", 17, "bold")
FONT_MONO  = ("Consolas", 9)

THUMB_SIZE = (280, 196)
WIN_WIDTH  = 560
WIN_HEIGHT = 720


def _to_displayable(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode in ("RGB", "RGBA", "L", "P"):
        return pil_img

    import numpy as np
    arr = np.array(pil_img, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr[:] = 0
    return Image.fromarray(arr.astype(np.uint8), mode="L")


class HoverButton(tk.Button):
    def __init__(self, master, bg_normal: str, bg_hover: str,
                 fg_normal: str = FG, **kw):
        super().__init__(
            master,
            bg=bg_normal,
            fg=fg_normal,
            activebackground=bg_hover,
            activeforeground=fg_normal,
            relief="flat",
            cursor="hand2",
            **kw,
        )
        self._bg_n = bg_normal
        self._bg_h = bg_hover

        self.bind("<Enter>", lambda _: self.config(bg=bg_hover))
        self.bind("<Leave>", lambda _: self.config(bg=bg_normal))


class StarCalibrationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._result: dict | None = None
        self._thumb_ref = None
        self._running = False

        self._build_window()
        self._build_ui()

    def _build_window(self):
        r = self.root
        r.title("Star Calibration")
        r.configure(bg=BG)
        r.resizable(True, True)
        r.minsize(420, 580)

        r.update_idletasks()
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        x = (sw - WIN_WIDTH)  // 2
        y = (sh - WIN_HEIGHT) // 2
        r.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{x}+{y}")

        style = ttk.Style(r)
        style.theme_use("default")
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=SURFACE,
            background=ACCENT,
            borderwidth=0,
            thickness=6,
        )
        style.configure("TSeparator", background=BORDER)

    def _build_ui(self):
        root = self.root

        hdr = tk.Frame(root, bg=SURFACE, pady=20)
        hdr.pack(fill="x")

        tk.Label(
            hdr, text="★  Star Calibration",
            font=FONT_TITLE, bg=SURFACE, fg=FG,
        ).pack()

        tk.Label(
            hdr,
            text="Calibrate a GONet all-sky image using the Gaia star catalogue",
            font=("Segoe UI", 9), bg=SURFACE, fg=FG_DIM,
        ).pack(pady=(4, 0))

        ttk.Separator(root).pack(fill="x")

        body = tk.Frame(root, bg=BG, padx=28, pady=20)
        body.pack(fill="both", expand=True)

        tk.Label(
            body, text="Image File", font=FONT_SB,
            bg=BG, fg=FG_DIM, anchor="w",
        ).pack(fill="x")

        file_row = tk.Frame(body, bg=BG)
        file_row.pack(fill="x", pady=(4, 14))

        self.file_var = tk.StringVar()

        file_entry = tk.Entry(
            file_row,
            textvariable=self.file_var,
            font=FONT_MONO, bg=SURFACE, fg=FG,
            insertbackground=FG,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        file_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        HoverButton(
            file_row, bg_normal=SURFACE, bg_hover=BORDER,
            text="Browse…", font=FONT_SB,
            command=self._select_file, padx=12, pady=5,
        ).pack(side="left")

        self.thumb_frame = tk.Frame(
            body, bg=SURFACE,
            width=THUMB_SIZE[0], height=THUMB_SIZE[1],
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.thumb_frame.pack_propagate(False)
        self.thumb_frame.pack(pady=(0, 18))

        self.thumb_label = tk.Label(
            self.thumb_frame, bg=SURFACE, fg=FG_DIM,
            font=FONT, text="No image selected",
        )
        self.thumb_label.place(relx=0.5, rely=0.5, anchor="center")

        # ── Detection parameters ─────────────────────────────────────────────
        params_row = tk.Frame(body, bg=BG)
        params_row.pack(fill="x", pady=(0, 14))

        tk.Label(params_row, text="Detection threshold (N σ):",
                 font=FONT, bg=BG, fg=FG_DIM).pack(side="left")
        self.n_var = tk.DoubleVar(value=5.0)
        tk.Spinbox(
            params_row, from_=1.0, to=20.0, increment=0.5,
            textvariable=self.n_var, width=5,
            font=FONT, bg=SURFACE, fg=FG,
            buttonbackground=BORDER, relief="flat",
            insertbackground=FG,
        ).pack(side="left", padx=(4, 20))

        tk.Label(params_row, text="Gaia mag limit:",
                 font=FONT, bg=BG, fg=FG_DIM).pack(side="left")
        self.gmax_var = tk.DoubleVar(value=2.5)
        tk.Spinbox(
            params_row, from_=1.0, to=8.0, increment=0.5,
            textvariable=self.gmax_var, width=5,
            font=FONT, bg=SURFACE, fg=FG,
            buttonbackground=BORDER, relief="flat",
            insertbackground=FG,
        ).pack(side="left", padx=(4, 0))

        # ── Options row ───────────────────────────────────────────────────
        opts_row = tk.Frame(body, bg=BG)
        opts_row.pack(fill="x", pady=(0, 6))

        self.show_plots_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            opts_row, text="Show diagnostic plots after run",
            variable=self.show_plots_var,
            font=FONT, bg=BG, fg=FG_DIM,
            activebackground=BG, activeforeground=FG,
            selectcolor=SURFACE,
        ).pack(side="left")

        self.run_btn = HoverButton(
            body, bg_normal=ACCENT, bg_hover=ACCENT_H,
            fg_normal=BG,
            text="▶  Run Calibration",
            font=FONT_LG,
            command=self._start_calibration,
            padx=24, pady=10,
        )
        self.run_btn.pack(pady=(0, 16))

        self.progress = ttk.Progressbar(
            body, mode="indeterminate",
            style="Dark.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(0, 6))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            body, textvariable=self.status_var,
            font=("Segoe UI", 9), bg=BG, fg=FG_DIM,
        ).pack()

        ttk.Separator(body).pack(fill="x", pady=16)

        self.results_outer = tk.Frame(body, bg=BG)

        tk.Label(
            self.results_outer, text="Calibration Results",
            font=FONT_SB, bg=BG, fg=FG_DIM, anchor="w",
        ).pack(fill="x", pady=(0, 8))

        self.results_box = tk.Frame(
            self.results_outer, bg=SURFACE,
            highlightthickness=1, highlightbackground=BORDER,
            padx=16, pady=12,
        )
        self.results_box.pack(fill="x")

        self.result_labels: dict = {}
        for key, display_name in [
            ("score", "Match score"),
            ("rms",   "RMS error"),
            ("wcs",   "WCS fit"),
            ("stars", "WCS stars"),
            ("shift", "Applied shift"),
        ]:
            row = tk.Frame(self.results_box, bg=SURFACE)
            row.pack(fill="x", pady=2)

            tk.Label(
                row, text=f"{display_name}:",
                font=FONT, bg=SURFACE, fg=FG_DIM,
                width=14, anchor="w",
            ).pack(side="left")

            lbl = tk.Label(row, text="—", font=FONT_MONO, bg=SURFACE, fg=FG, anchor="w")
            lbl.pack(side="left")
            self.result_labels[key] = lbl

        btn_row = tk.Frame(self.results_outer, bg=BG)
        btn_row.pack(fill="x", pady=(10, 0))

        HoverButton(
            btn_row, bg_normal=SURFACE, bg_hover=BORDER,
            text="★  View Matched Stars",
            font=FONT_SB, padx=16, pady=7,
            command=self._open_matched_stars,
        ).pack(side="left")

        HoverButton(
            btn_row, bg_normal=SURFACE, bg_hover=BORDER,
            text="🗺  Star Map",
            font=FONT_SB, padx=16, pady=7,
            command=self._open_star_map,
        ).pack(side="left", padx=(8, 0))

        self.save_btn = HoverButton(
            btn_row,
            bg_normal=SURFACE, bg_hover=BORDER,
            text="⬇  Save Shifted Image",
            font=FONT_SB, padx=16, pady=7,
            command=self._save_shifted_image,
        )
        self.save_btn.pack(side="right")

    def _select_file(self):
        path = filedialog.askopenfilename(
            title="Select GONet Image",
            filetypes=[
                ("All Files",  "*.*"),
                ("JPG Files",  "*.jpg"),
                ("JPEG Files", "*.jpeg"),
                ("PNG Files",  "*.png"),
                ("TIFF Files", "*.tif;*.tiff"),
            ],
        )
        if not path:
            return

        self.file_var.set(path)
        self._load_thumbnail(path)
        self._hide_results()
        self.status_var.set("Ready")

    def _load_thumbnail(self, path: str):
        try:
            img   = _to_displayable(Image.open(path))
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self._thumb_ref = photo
            self.thumb_label.config(image=photo, text="")
        except Exception:
            self.thumb_label.config(image="", text="Preview unavailable")

    def _start_calibration(self):
        path = self.file_var.get().strip()
        if not path:
            messagebox.showwarning("No file selected",
                                   "Please select a GONet image file first.")
            return

        if self._running:
            return

        self._running = True
        self._result  = None

        self.run_btn.config(state="disabled", text="  Processing…")
        self.progress.start(12)
        self.status_var.set("Running calibration — this may take a minute…")
        self._hide_results()

        try:
            N    = float(self.n_var.get())
            gmax = float(self.gmax_var.get())
        except (tk.TclError, ValueError):
            N, gmax = 5.0, 2.5

        show_plots = bool(self.show_plots_var.get())
        threading.Thread(target=self._worker, args=(path, N, gmax, show_plots), daemon=True).start()

    def _worker(self, path: str, N: float, gmax: float, show_plots: bool):
        try:
            result = run_calibration(path, show_plots=False, N=N, gmax=gmax)
            result["_show_plots"] = show_plots
            self.root.after(0, lambda: self._on_success(result))
        except Exception as exc:
            msg = str(exc)
            self.root.after(0, lambda m=msg: self._on_error(m))

    def _on_success(self, result: dict):
        self._result  = result
        self._running = False

        self.progress.stop()
        self.run_btn.config(state="normal", text="▶  Run Calibration")
        self.status_var.set("Calibration complete.")

        best       = result["best"]
        wcs_result = result["wcs_result"]

        rms_text = (
            f"{best['rms_pix']:.2f} px"
            if not math.isnan(best["rms_pix"])
            else "n/a"
        )

        wcs_ok     = wcs_result["wcs_fit_success"]
        wcs_text   = "Success" if wcs_ok else "Failed"
        shift_text = (
            f"dx={wcs_result['shift_x']:+.1f}  dy={wcs_result['shift_y']:+.1f} px"
            if wcs_ok else "n/a"
        )

        self.result_labels["score"].config(
            text=f"{best['score']} matches",
            fg=SUCCESS if best["score"] > 5 else WARN,
        )
        self.result_labels["rms"].config(text=rms_text)
        self.result_labels["wcs"].config(
            text=wcs_text, fg=SUCCESS if wcs_ok else ERROR,
        )
        self.result_labels["stars"].config(text=str(wcs_result["n_wcs_matches"]))
        self.result_labels["shift"].config(text=shift_text)

        self._show_results()
        self._open_preview(result)
        if result.get("_show_plots"):
            self._show_diagnostic_plots(result)

    def _show_diagnostic_plots(self, result: dict):
        import numpy as np
        import matplotlib.pyplot as plt
        best       = result["best"]
        wcs_result = result["wcs_result"]
        sub        = result["sub"]
        img_xy     = result["img_xy"]
        sub_mean   = float(np.mean(sub))
        sub_std    = float(np.std(sub))

        plt.figure()
        plt.imshow(sub, origin="lower", cmap="gray",
                   vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
        plt.scatter(img_xy[:, 0], img_xy[:, 1],
                    s=50, edgecolor="red", facecolor="none", label="Image sources")
        plt.scatter(best["pred_xy"][:, 0], best["pred_xy"][:, 1],
                    s=50, edgecolor="blue", facecolor="none", label="Predicted sources")
        plt.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]],
                    s=100, marker="+", c="yellow", label="Target center")
        if wcs_result["wcs_fit_success"]:
            plt.scatter([wcs_result["zenith_x"]], [wcs_result["zenith_y"]],
                        s=120, marker="x", c="cyan", label="Zenith (WCS)")
            plt.plot(
                [wcs_result["zenith_x"], wcs_result["target_cx"]],
                [wcs_result["zenith_y"], wcs_result["target_cy"]],
                color="cyan", linestyle="--", linewidth=1.5, label="Applied shift",
            )
        plt.legend()
        plt.title(f"Best score: {best['score']} matches")
        plt.show(block=False)

        plt.figure()
        plt.imshow(wcs_result["centered_sub"], origin="lower", cmap="gray",
                   vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
        plt.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]],
                    s=120, marker="x", c="cyan", label="Centered zenith target")
        plt.legend()
        plt.title("Image shifted so zenith is centered")
        plt.show(block=False)

    def _on_error(self, msg: str):
        self._running = False
        self.progress.stop()
        self.run_btn.config(state="normal", text="▶  Run Calibration")
        self.status_var.set("Calibration failed.")
        messagebox.showerror("Calibration Failed", msg)

    def _show_results(self):
        self.results_outer.pack(fill="x")

    def _hide_results(self):
        self.results_outer.pack_forget()

    def _open_preview(self, result: dict):
        shifted_image  = result.get("shifted_image")
        shifted_format = result.get("shifted_format", "PNG")
        suggested_ext  = result.get("suggested_suffix", ".png")

        if shifted_image is None:
            messagebox.showinfo(
                "No preview",
                "Calibration finished but no shifted image was returned.",
            )
            return

        win = tk.Toplevel(self.root)
        win.title("Shifted Image Preview")
        win.configure(bg=BG)
        win.resizable(True, True)

        hdr = tk.Frame(win, bg=SURFACE, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Calibrated Image Preview",
                 font=FONT_LG, bg=SURFACE, fg=FG).pack()

        preview = _to_displayable(shifted_image.copy())
        preview.thumbnail((1000, 700), Image.LANCZOS)

        photo     = ImageTk.PhotoImage(preview)
        img_label = tk.Label(win, image=photo, bg=BG)
        img_label.image = photo
        img_label.pack(padx=16, pady=12)

        btn_bar = tk.Frame(win, bg=BG)
        btn_bar.pack(pady=(0, 14))

        def _save():
            save_path = filedialog.asksaveasfilename(
                title="Save Shifted Image",
                defaultextension=suggested_ext,
                filetypes=[
                    ("JPEG Files", "*.jpg;*.jpeg"),
                    ("PNG Files",  "*.png"),
                    ("TIFF Files", "*.tif;*.tiff"),
                    ("All Files",  "*.*"),
                ],
            )
            if not save_path:
                return
            try:
                shifted_image.save(save_path, format=shifted_format)
                messagebox.showinfo("Saved", f"Shifted image saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Save Failed", str(e))

        HoverButton(
            btn_bar, bg_normal=ACCENT, bg_hover=ACCENT_H,
            fg_normal=BG, text="⬇  Save Image",
            font=FONT_SB, padx=16, pady=6,
            command=_save,
        ).pack(side="left", padx=6)

        HoverButton(
            btn_bar, bg_normal=SURFACE, bg_hover=BORDER,
            text="Close", font=FONT_SB, padx=16, pady=6,
            command=win.destroy,
        ).pack(side="left", padx=6)

    def _open_matched_stars(self):
        if self._result is None:
            return
        stars = self._result.get("matched_stars", [])

        win = tk.Toplevel(self.root)
        win.title(f"Matched Stars ({len(stars)})")
        win.configure(bg=BG)
        win.resizable(True, True)

        hdr = tk.Frame(win, bg=SURFACE, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"Matched Stars  (× {len(stars)})",
                 font=FONT_LG, bg=SURFACE, fg=FG).pack()

        # scrollable Treeview
        frame = tk.Frame(win, bg=BG)
        frame.pack(fill="both", expand=True, padx=16, pady=12)

        cols = ("#", "Alt (°)", "Az (°)", "G mag", "Img X", "Img Y", "Dist (px)")
        style = ttk.Style(win)
        style.theme_use("default")
        style.configure("Stars.Treeview",
                        background=SURFACE, foreground=FG,
                        fieldbackground=SURFACE, rowheight=22,
                        font=FONT_MONO)
        style.configure("Stars.Treeview.Heading",
                        background=BORDER, foreground=FG, font=FONT_SB)
        style.map("Stars.Treeview", background=[("selected", ACCENT)])

        tree = ttk.Treeview(frame, columns=cols, show="headings",
                            style="Stars.Treeview", height=20)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="center")
        tree.column("#", width=40)

        for i, s in enumerate(sorted(stars, key=lambda x: x["gmag"]), 1):
            tree.insert("", "end", values=(
                i,
                f"{s['alt']:.3f}",
                f"{s['az']:.3f}",
                f"{s['gmag']:.2f}",
                f"{s['img_x']:.1f}",
                f"{s['img_y']:.1f}",
                f"{s['dist_px']:.2f}",
            ))

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        HoverButton(
            win, bg_normal=SURFACE, bg_hover=BORDER,
            text="Close", font=FONT_SB, padx=16, pady=6,
            command=win.destroy,
        ).pack(pady=(0, 12))

    def _open_star_map(self):
        if self._result is None:
            return
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import astropy.units as u
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, AltAz, SkyCoord

        # ── Bright-star name catalog (ra_deg, dec_deg, name) ──────────────
        BRIGHT_STARS = [
            (101.287, -16.716, "Sirius"),       (95.988,  -52.696, "Canopus"),
            (213.915,  19.182, "Arcturus"),      (279.235,  38.784, "Vega"),
            (79.172,  45.998,  "Capella"),       (78.634,  -8.202,  "Rigel"),
            (114.826,  5.225,  "Procyon"),       (24.429, -57.237,  "Achernar"),
            (88.793,   7.407,  "Betelgeuse"),    (210.956, -60.373, "Hadar"),
            (297.696,   8.868, "Altair"),        (186.650, -63.099, "Acrux"),
            (68.980,   16.509, "Aldebaran"),     (247.352, -26.432, "Antares"),
            (201.298,  -11.161,"Spica"),         (116.329,  28.026, "Pollux"),
            (344.413,  -29.622,"Fomalhaut"),     (310.358,  45.280, "Deneb"),
            (191.930,  -59.689,"Mimosa"),        (152.093,  11.967, "Regulus"),
            (104.656,  -28.972,"Adhara"),        (113.649,  31.888, "Castor"),
            (263.402,  -37.104,"Shaula"),        (187.791, -57.113, "Gacrux"),
            (84.053,   21.143, "Elnath"),        (81.573,   6.350,  "Alnilam"),
            (165.932,  61.751, "Alioth"),        (193.507,  55.960, "Dubhe"),
            (206.885,  49.313, "Alkaid"),        (177.265,  14.572, "Denebola"),
            (55.730,   47.788, "Mirfak"),        (10.897,   56.537, "Schedar"),
            (37.955,   89.264, "Polaris"),       (45.570,   4.090,  "Menkar"),
            (346.190,  15.205, "Enif"),          (322.165,  46.741, "Sador"),
            (138.300,  -69.717,"Avior"),         (125.628, -59.510, "Naos"),
            (239.713,  26.714, "Izar"),
        ]
        # Build SkyCoord array for fast batch separation
        _cat_ra  = np.array([s[0] for s in BRIGHT_STARS])
        _cat_dec = np.array([s[1] for s in BRIGHT_STARS])
        _cat_sc  = SkyCoord(ra=_cat_ra * u.deg, dec=_cat_dec * u.deg, frame="icrs")

        def _star_name(alt_deg, az_deg, meta, tol_deg=1.5):
            loc   = EarthLocation(
                lat=float(meta["GPS"]["latitude"]) * u.deg,
                lon=float(meta["GPS"]["longitude"]) * u.deg,
                height=float(meta["GPS"]["altitude"]) * u.m,
            )
            t     = Time(meta["DateTime"].replace(":", "-", 2).replace(" ", "T"), scale="utc")
            frame = AltAz(obstime=t, location=loc)
            sc    = SkyCoord(alt=alt_deg * u.deg, az=az_deg * u.deg, frame=frame).icrs
            seps  = sc.separation(_cat_sc).deg
            best  = int(np.argmin(seps))
            return BRIGHT_STARS[best][2] if seps[best] < tol_deg else None

        # ── Plot ──────────────────────────────────────────────────────────
        result     = self._result
        stars      = result.get("matched_stars", [])
        img_xy     = result["img_xy"]
        wcs_result = result["wcs_result"]
        meta       = result.get("meta")
        dx         = wcs_result["shift_x"]
        dy         = wcs_result["shift_y"]
        img        = wcs_result["centered_sub"]

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        m, s = img.mean(), img.std()
        ax.imshow(img, origin="upper", cmap="gray",
                  vmin=m - 2 * s, vmax=m + 6 * s)

        # All detected sources
        sx = img_xy[:, 0] + dx
        sy = img_xy[:, 1] + dy
        ax.scatter(sx, sy, s=30, edgecolors="#f85149", facecolors="none",
                   linewidths=0.8, label=f"All sources ({len(img_xy)})")

        # Matched stars with name labels
        for star in sorted(stars, key=lambda x: x["gmag"]):
            mx = star["img_x"] + dx
            my = star["img_y"] + dy
            ax.scatter(mx, my, s=120, edgecolors="#58a6ff", facecolors="none",
                       linewidths=1.5)
            name = _star_name(star["alt"], star["az"], meta) if meta else None
            label = name if name else f"G{star['gmag']:.1f}"
            ax.annotate(
                label,
                xy=(mx, my), xytext=(6, -10),
                textcoords="offset points",
                color="#79b8ff", fontsize=7,
                fontfamily="monospace",
            )

        matched_patch = mpatches.Circle((0, 0), radius=1, edgecolor="#58a6ff",
                                        facecolor="none", linewidth=1.5,
                                        label=f"Matched ({len(stars)})")
        ax.legend(handles=[ax.get_legend_handles_labels()[0][0], matched_patch],
                  facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
                  fontsize=8)
        ax.set_title(f"Star Map — {len(stars)} matched  |  rms = {result['best']['rms_pix']:.2f} px",
                     color="#e6edf3", fontsize=11)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        plt.tight_layout()
        plt.show(block=False)

    def _save_shifted_image(self):
        if self._result is None:
            return
        self._open_preview(self._result)


root = tk.Tk()
app  = StarCalibrationApp(root)
root.mainloop()
