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
            text="Zenith-centre a GONet all-sky image using the Gaia star catalogue",
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

        self.save_btn = HoverButton(
            self.results_outer,
            bg_normal=SURFACE, bg_hover=BORDER,
            text="⬇  Save Shifted Image",
            font=FONT_SB, padx=16, pady=7,
            command=self._save_shifted_image,
        )
        self.save_btn.pack(pady=(10, 0), anchor="e")

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

        threading.Thread(target=self._worker, args=(path,), daemon=True).start()

    def _worker(self, path: str):
        try:
            result = run_calibration(path, show_plots=False)
            self.root.after(0, lambda: self._on_success(result))
        except Exception as exc:
            msg = str(exc)
            self.root.after(0, lambda: self._on_error(msg))

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

    def _save_shifted_image(self):
        if self._result is None:
            return
        self._open_preview(self._result)


root = tk.Tk()
app  = StarCalibrationApp(root)
root.mainloop()
