"""
starcalibration.py — Star Calibration GUI
==========================================
A Tkinter desktop application that lets you pick a GONet all-sky camera
image, run the star-calibration pipeline (star detection → Gaia catalogue
download → orientation solving → WCS fitting → zenith centring), preview the
result, and save the calibrated image.

Design goals
------------
* Dark "space" colour palette so the star imagery looks natural.
* Non-blocking: the heavy maths runs on a background daemon thread so the
  window stays responsive and shows a progress bar while waiting.
* Supports any bit-depth that Pillow can open (8-bit JPG/PNG, 16-bit TIFF, …).

Layout overview (top → bottom)
-------------------------------
  ┌─ Header (title + subtitle) ──────────────────────────────────────────────┐
  ├─ Body ───────────────────────────────────────────────────────────────────┤
  │   File path entry + Browse button                                        │
  │   Thumbnail preview (280 × 196)                                          │
  │   ▶ Run Calibration button                                               │
  │   Progress bar + status text                                             │
  │   ─────────────────────────────────────────────                          │
  │   Results panel (shown only after success)                               │
  │     match score / RMS / WCS status / star count / shift                  │
  │     ⬇ Save Shifted Image button                                          │
  └──────────────────────────────────────────────────────────────────────────┘
"""

import math           # used for math.isnan() to check if RMS is not-a-number
import threading      # runs the calibration on a background thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from stellarcalibrationbackend import run_calibration

# ─── Palette ─────────────────────────────────────────────────────────────────
BG       = "#0d1117"
SURFACE  = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
ACCENT_H = "#79b8ff"   # hover
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

THUMB_SIZE = (280, 196)       # thumbnail preview dimensions
WIN_WIDTH  = 560
WIN_HEIGHT = 720


# =============================================================================
# HELPER: convert any PIL image to an 8-bit greyscale for Tkinter display
# =============================================================================

def _to_displayable(pil_img: Image.Image) -> Image.Image:
    """
    Convert any PIL image to an 8-bit format that ImageTk.PhotoImage can show.

    Why is this needed?
    -------------------
    Tkinter's ImageTk only accepts 8-bit images (modes "L", "RGB", "RGBA", "P").
    TIFF files can be 16-bit ("I;16"), 32-bit integer ("I"), or 32-bit float
    ("F"), which would crash ImageTk.  This function handles all those cases
    by stretching the pixel values to the 0–255 range and returning an 8-bit
    greyscale ("L") image.

    Parameters
    ----------
    pil_img : PIL Image
        Any PIL image, regardless of mode or bit-depth.

    Returns
    -------
    PIL Image in a mode that ImageTk.PhotoImage accepts.
    """
    # If the image is already in a Tkinter-compatible mode, return it unchanged.
    if pil_img.mode in ("RGB", "RGBA", "L", "P"):
        return pil_img

    # For high-bit-depth images: convert to float array, stretch to [0, 255],
    # then convert back to an 8-bit greyscale PIL Image.
    import numpy as np
    arr = np.array(pil_img, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        # Normalise: map the darkest pixel to 0 and the brightest to 255.
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        # Flat image (all pixels identical) — set everything to black.
        arr[:] = 0
    return Image.fromarray(arr.astype(np.uint8), mode="L")


# =============================================================================
# WIDGET: HoverButton — a Tk Button that changes colour on mouse-over
# =============================================================================

class HoverButton(tk.Button):
    """
    A standard tkinter Button with a simple hover effect.

    When the user moves the mouse over this button it changes to bg_hover,
    and reverts to bg_normal when the mouse leaves.  This gives the UI a
    more polished feel without requiring any external library.
    """

    def __init__(self, master, bg_normal: str, bg_hover: str,
                 fg_normal: str = FG, **kw):
        """
        Parameters
        ----------
        master    : parent widget
        bg_normal : background colour when the mouse is NOT over the button
        bg_hover  : background colour when the mouse IS over the button
        fg_normal : text colour (default: FG off-white)
        **kw      : any additional keyword arguments passed to tk.Button
        """
        super().__init__(
            master,
            bg=bg_normal,
            fg=fg_normal,
            activebackground=bg_hover,
            activeforeground=fg_normal,
            relief="flat",       # remove the default 3-D border
            cursor="hand2",      # show a pointer cursor on hover
            **kw,
        )
        # Store both colours so the event bindings can switch between them.
        self._bg_n = bg_normal
        self._bg_h = bg_hover

        # Bind mouse-enter / mouse-leave events to change the background colour.
        self.bind("<Enter>", lambda _: self.config(bg=bg_hover))
        self.bind("<Leave>", lambda _: self.config(bg=bg_normal))


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class StarCalibrationApp:
    """
    Tkinter application for the GONet star-calibration tool.

    Responsibilities
    ----------------
    * Build and configure the main window and all child widgets.
    * Handle user interactions (file selection, run button click, save).
    * Launch the calibration pipeline on a background thread and relay
      results back to the GUI thread safely via root.after().
    * Display results and open a preview window for the calibrated image.
    """

    def __init__(self, root: tk.Tk):
        self.root = root

        # Stores the full result dict returned by run_calibration().
        # Used by _save_shifted_image() to re-open the preview without
        # re-running the pipeline.
        self._result: dict | None = None

        # Holds a reference to the thumbnail PhotoImage so that Python's
        # garbage collector does not delete it while Tkinter is still
        # displaying it (a common Tkinter gotcha).
        self._thumb_ref = None

        # Flag that prevents the user from clicking Run a second time while
        # a calibration is already in progress.
        self._running = False

        # Build the window chrome, then populate all the widgets.
        self._build_window()
        self._build_ui()

    # =========================================================================
    # WINDOW SETUP
    # =========================================================================

    def _build_window(self):
        """
        Configure the root Tk window: title, size, position, and ttk styles.

        The window is centred on screen and made resizable so the user can
        enlarge it on bigger monitors.
        """
        r = self.root
        r.title("Star Calibration")
        r.configure(bg=BG)

        # Allow the user to drag the window edges to resize it.
        # minsize prevents the window from becoming too small to use.
        r.resizable(True, True)
        r.minsize(420, 580)

        # Centre the window on screen.
        r.update_idletasks()   # force Tk to calculate window dimensions
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        x = (sw - WIN_WIDTH)  // 2
        y = (sh - WIN_HEIGHT) // 2
        r.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{x}+{y}")

        # Apply custom styles to ttk widgets (progress bar and separator)
        # so they match the dark colour palette.  ttk widgets do not respond
        # to the standard bg/fg options, so we use ttk.Style instead.
        style = ttk.Style(r)
        style.theme_use("default")
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=SURFACE,   # the empty track of the progress bar
            background=ACCENT,     # the filled portion
            borderwidth=0,
            thickness=6,           # thin, modern-looking bar
        )
        style.configure("TSeparator", background=BORDER)

    # =========================================================================
    # UI CONSTRUCTION
    # =========================================================================

    def _build_ui(self):
        """
        Create and lay out every widget in the main window.

        Uses pack() geometry manager throughout.  Widgets are grouped into:
          - hdr  : top header bar (title + subtitle)
          - body : the main content area
        """
        root = self.root

        # ── Header band ─────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=SURFACE, pady=20)
        hdr.pack(fill="x")   # stretch across the full window width

        tk.Label(
            hdr, text="★  Star Calibration",
            font=FONT_TITLE, bg=SURFACE, fg=FG,
        ).pack()

        tk.Label(
            hdr,
            text="Zenith-centre a GONet all-sky image using the Gaia star catalogue",
            font=("Segoe UI", 9), bg=SURFACE, fg=FG_DIM,
        ).pack(pady=(4, 0))

        # Thin horizontal divider to separate the header from the body.
        ttk.Separator(root).pack(fill="x")

        # ── Body frame ───────────────────────────────────────────────────────
        body = tk.Frame(root, bg=BG, padx=28, pady=20)
        body.pack(fill="both", expand=True)

        # ── File selection ──────────────────────────────────────────────────
        tk.Label(
            body, text="Image File", font=FONT_SB,
            bg=BG, fg=FG_DIM, anchor="w",
        ).pack(fill="x")

        # A single row containing the path entry and the Browse button.
        file_row = tk.Frame(body, bg=BG)
        file_row.pack(fill="x", pady=(4, 14))

        # StringVar allows us to read/write the path text programmatically.
        self.file_var = tk.StringVar()

        # Text entry where the selected file path is shown.
        file_entry = tk.Entry(
            file_row,
            textvariable=self.file_var,
            font=FONT_MONO, bg=SURFACE, fg=FG,
            insertbackground=FG,      # cursor colour inside the field
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,    # coloured border when the entry has focus
        )
        file_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        HoverButton(
            file_row, bg_normal=SURFACE, bg_hover=BORDER,
            text="Browse…", font=FONT_SB,
            command=self._select_file, padx=12, pady=5,
        ).pack(side="left")

        # ── Thumbnail preview ────────────────────────────────────────────────
        # A fixed-size box that displays a small preview of the selected image.
        self.thumb_frame = tk.Frame(
            body, bg=SURFACE,
            width=THUMB_SIZE[0], height=THUMB_SIZE[1],
            highlightthickness=1, highlightbackground=BORDER,
        )
        self.thumb_frame.pack_propagate(False)  # keep the fixed size
        self.thumb_frame.pack(pady=(0, 18))

        # Placeholder text shown before an image is selected.
        self.thumb_label = tk.Label(
            self.thumb_frame, bg=SURFACE, fg=FG_DIM,
            font=FONT, text="No image selected",
        )
        self.thumb_label.place(relx=0.5, rely=0.5, anchor="center")

        # ── Run button ───────────────────────────────────────────────────────
        self.run_btn = HoverButton(
            body, bg_normal=ACCENT, bg_hover=ACCENT_H,
            fg_normal=BG,            # dark text on the blue button
            text="▶  Run Calibration",
            font=FONT_LG,
            command=self._start_calibration,
            padx=24, pady=10,
        )
        self.run_btn.pack(pady=(0, 16))

        # ── Progress bar and status line ─────────────────────────────────────
        # The progress bar uses "indeterminate" mode (bouncing animation)
        # because we don't know how long the calibration will take.
        self.progress = ttk.Progressbar(
            body, mode="indeterminate",
            style="Dark.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(0, 6))

        # A single line of text below the bar that describes the current state.
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            body, textvariable=self.status_var,
            font=("Segoe UI", 9), bg=BG, fg=FG_DIM,
        ).pack()

        # Divider before the results section.
        ttk.Separator(body).pack(fill="x", pady=16)

        # ── Results panel ────────────────────────────────────────────────────
        # This whole frame is hidden (pack_forget) until calibration succeeds,
        # then revealed via _show_results().
        self.results_outer = tk.Frame(body, bg=BG)

        tk.Label(
            self.results_outer, text="Calibration Results",
            font=FONT_SB, bg=BG, fg=FG_DIM, anchor="w",
        ).pack(fill="x", pady=(0, 8))

        # A card-style box with a subtle border for the result key-value pairs.
        self.results_box = tk.Frame(
            self.results_outer, bg=SURFACE,
            highlightthickness=1, highlightbackground=BORDER,
            padx=16, pady=12,
        )
        self.results_box.pack(fill="x")

        # Build one row per result metric.  Each row has a dim label on the
        # left and a dynamic value label on the right.
        # We store the value labels in a dict so _on_success() can update them.
        self.result_labels: dict = {}
        for key, display_name in [
            ("score", "Match score"),   # number of catalog stars matched to image stars
            ("rms",   "RMS error"),     # root-mean-square pixel distance of matches
            ("wcs",   "WCS fit"),       # whether the WCS (sky projection) fit succeeded
            ("stars", "WCS stars"),     # how many stars were used to fit the WCS
            ("shift", "Applied shift"), # pixel translation applied to centre the zenith
        ]:
            row = tk.Frame(self.results_box, bg=SURFACE)
            row.pack(fill="x", pady=2)

            # Left column: static metric name (dim colour)
            tk.Label(
                row, text=f"{display_name}:",
                font=FONT, bg=SURFACE, fg=FG_DIM,
                width=14, anchor="w",
            ).pack(side="left")

            # Right column: dynamic value (updated after calibration)
            lbl = tk.Label(row, text="—", font=FONT_MONO, bg=SURFACE, fg=FG, anchor="w")
            lbl.pack(side="left")
            self.result_labels[key] = lbl

        # Save button sits below the result rows.
        self.save_btn = HoverButton(
            self.results_outer,
            bg_normal=SURFACE, bg_hover=BORDER,
            text="⬇  Save Shifted Image",
            font=FONT_SB, padx=16, pady=7,
            command=self._save_shifted_image,
        )
        self.save_btn.pack(pady=(10, 0), anchor="e")

    # =========================================================================
    # USER ACTIONS
    # =========================================================================

    def _select_file(self):
        """
        Open a file-picker dialog, store the selected path, and load a thumbnail.

        Supports GONet proprietary files as well as standard JPG, PNG, and TIFF.
        The results panel is hidden immediately so the user starts fresh if they
        pick a different image after a previous calibration.
        """
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
            return   # user cancelled the dialog — do nothing

        self.file_var.set(path)
        self._load_thumbnail(path)
        self._hide_results()
        self.status_var.set("Ready")

    def _load_thumbnail(self, path: str):
        """
        Open the image at `path` and display a small preview in thumb_label.

        _to_displayable() is called first so 16-bit TIFFs render correctly.
        PIL's thumbnail() resizes in-place without distorting the aspect ratio.
        """
        try:
            img   = _to_displayable(Image.open(path))
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)

            # PhotoImage must be stored in self._thumb_ref or Python's garbage
            # collector will delete the object immediately, causing the label
            # to show a blank area.
            photo = ImageTk.PhotoImage(img)
            self._thumb_ref = photo
            self.thumb_label.config(image=photo, text="")
        except Exception:
            # If Pillow can't open a file it raises an exception.  Rather than
            # crashing, show a graceful "unavailable" message.
            self.thumb_label.config(image="", text="Preview unavailable")

    def _start_calibration(self):
        """
        Validate input, update the UI to "running" state, and launch the
        background calibration thread.

        Why a background thread?
        Because run_calibration() can take over a minute (it queries the
        Gaia catalogue over the internet and then solves thousands of angle
        combinations).  If we ran it on the main thread the window would
        freeze completely — no progress bar, no ability to move the window.
        daemon=True means the thread dies automatically if the main window
        is closed, so we never leave a zombie process.
        """
        path = self.file_var.get().strip()
        if not path:
            messagebox.showwarning("No file selected",
                                   "Please select a GONet image file first.")
            return

        if self._running:
            return   # already running — ignore extra clicks

        self._running = True
        self._result  = None

        # Disable the run button and show a progress bar.
        self.run_btn.config(state="disabled", text="  Processing…")
        self.progress.start(12)   # speed: update every 12 ms
        self.status_var.set("Running calibration — this may take a minute…")
        self._hide_results()

        # Start the worker on a background (daemon) thread.
        threading.Thread(target=self._worker, args=(path,), daemon=True).start()

    def _worker(self, path: str):
        """
        Worker function that runs on the background thread.

        Calls run_calibration(), then schedules the appropriate callback on the
        main (GUI) thread via root.after().  Direct Tk widget calls from a
        background thread are NOT safe — root.after() is the correct mechanism.
        """
        try:
            result = run_calibration(path, show_plots=False)
            # Schedule success callback on the GUI thread.
            self.root.after(0, lambda: self._on_success(result))
        except Exception as exc:
            msg = str(exc)
            # Schedule error callback on the GUI thread.
            self.root.after(0, lambda: self._on_error(msg))

    def _on_success(self, result: dict):
        """
        Called on the main thread when calibration finishes successfully.

        Stops the progress bar, populates the results panel, and opens the
        calibrated-image preview window.
        """
        self._result  = result
        self._running = False

        # Stop the progress animation and restore the run button.
        self.progress.stop()
        self.run_btn.config(state="normal", text="▶  Run Calibration")
        self.status_var.set("Calibration complete.")

        best       = result["best"]
        wcs_result = result["wcs_result"]

        # --- Format the metric values for display ---
        # math.isnan() checks whether the RMS value is "not a number" (which
        # happens when there were zero matched stars, making RMS undefined).
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

        # Update each label with the computed value; also colour-code a few of
        # them green/amber/red so the user gets an instant quality signal.
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
        """
        Called on the main thread when the calibration raises an exception.

        Stops the progress animation and shows an error dialog with the
        exception message.
        """
        self._running = False
        self.progress.stop()
        self.run_btn.config(state="normal", text="▶  Run Calibration")
        self.status_var.set("Calibration failed.")
        messagebox.showerror("Calibration Failed", msg)

    def _show_results(self):
        """Make the results panel visible by packing it into the layout."""
        self.results_outer.pack(fill="x")

    def _hide_results(self):
        """Remove the results panel from the layout (without destroying it)."""
        self.results_outer.pack_forget()

    # =========================================================================
    # PREVIEW WINDOW
    # =========================================================================

    def _open_preview(self, result: dict):
        """
        Open a separate Toplevel window showing the calibrated (shifted) image.

        The window contains the image and two buttons: Save and Close.
        Saving triggers a file-picker and calls PIL Image.save().

        The image is converted to 8-bit via _to_displayable() before being
        passed to ImageTk so that 16-bit TIFFs render correctly even though
        the preview is only 8-bit; the full-bit-depth original is what gets
        saved.
        """
        shifted_image  = result.get("shifted_image")
        shifted_format = result.get("shifted_format", "PNG")
        suggested_ext  = result.get("suggested_suffix", ".png")

        if shifted_image is None:
            messagebox.showinfo(
                "No preview",
                "Calibration finished but no shifted image was returned.",
            )
            return

        # Create a new modal-style window.
        win = tk.Toplevel(self.root)
        win.title("Shifted Image Preview")
        win.configure(bg=BG)
        win.resizable(True, True)

        # Header band matching the main window style.
        hdr = tk.Frame(win, bg=SURFACE, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Calibrated Image Preview",
                 font=FONT_LG, bg=SURFACE, fg=FG).pack()

        # Convert to 8-bit for display, then shrink to fit a reasonable window.
        preview = _to_displayable(shifted_image.copy())
        preview.thumbnail((1000, 700), Image.LANCZOS)

        # Keep the PhotoImage alive by storing a reference on the label widget.
        photo     = ImageTk.PhotoImage(preview)
        img_label = tk.Label(win, image=photo, bg=BG)
        img_label.image = photo   # prevents garbage collection
        img_label.pack(padx=16, pady=12)

        # ── Button bar ───────────────────────────────────────────────────────
        btn_bar = tk.Frame(win, bg=BG)
        btn_bar.pack(pady=(0, 14))

        def _save():
            """Inner function: ask for a save path and write the image."""
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
                return   # user cancelled
            try:
                # Save the full-quality image (not the 8-bit preview).
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
        """
        Re-open the preview window from the last stored result.

        Called by the "⬇ Save Shifted Image" button in the main window's
        results panel, allowing the user to save after closing the auto-opened
        preview, without having to re-run the calibration.
        """
        if self._result is None:
            return   # no result yet — nothing to save
        self._open_preview(self._result)


# =============================================================================
# ENTRY POINT
# =============================================================================

# When Python runs this file directly (i.e. not imported as a module), create
# the Tk root window, instantiate the app, and start the event loop.
# root.mainloop() blocks until the user closes the window.

root = tk.Tk()
app  = StarCalibrationApp(root)
root.mainloop()
