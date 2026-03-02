from tkinter import Tk, Button, Entry, END, Toplevel, Label
from tkinter import filedialog, messagebox
from PIL import ImageTk

from star_calibration_backend import run_calibration

root = Tk()
root.title("Star Calibration GUI")

def select_file():
    file_path = filedialog.askopenfilename(title="Select GONet Images", filetypes=[("All Files", "*.*"), ("JPG Files", "*.jpg"), ("PNG Files", "*.png"), ("TIFF Files", "*.tif"), ("JPEG", "*.jpeg*")])
    if file_path:
        file_entry.delete(0, END)
        file_entry.insert(0, file_path)


def run_calibration_from_gui():
    file_path = file_entry.get().strip()
    if not file_path:
        messagebox.showwarning("Missing file", "Please select an image file first.")
        return

    try:
        result = run_calibration(file_path)
        shifted_image = result.get("shifted_image") if isinstance(result, dict) else None
        shifted_format = result.get("shifted_format", "PNG") if isinstance(result, dict) else "PNG"
        suggested_suffix = result.get("suggested_suffix", ".png") if isinstance(result, dict) else ".png"

        if shifted_image is None:
            messagebox.showinfo("Calibration complete", "Calibration finished, but no shifted image was returned.")
            return

        preview_window = Toplevel(root)
        preview_window.title("Shifted Image Preview")

        preview_image = shifted_image.copy()
        preview_image.thumbnail((1000, 700))
        tk_preview_image = ImageTk.PhotoImage(preview_image)

        image_label = Label(preview_window, image=tk_preview_image)
        image_label.image = tk_preview_image
        image_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        def download_shifted_image():
            save_path = filedialog.asksaveasfilename(
                title="Save Shifted Image",
                defaultextension=suggested_suffix,
                filetypes=[
                    ("JPEG Files", "*.jpg;*.jpeg"),
                    ("PNG Files", "*.png"),
                    ("TIFF Files", "*.tif;*.tiff"),
                    ("All Files", "*.*"),
                ],
            )
            if not save_path:
                return

            try:
                shifted_image.save(save_path, format=shifted_format)
                messagebox.showinfo("Saved", f"Shifted image saved to:\n{save_path}")
            except Exception as save_error:
                messagebox.showerror("Save failed", str(save_error))

        download_button = Button(preview_window, text="Download Shifted Image", command=download_shifted_image)
        download_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        close_button = Button(preview_window, text="Close", command=preview_window.destroy)
        close_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
    except Exception as error:
        messagebox.showerror("Calibration failed", str(error))

button = Button(root, text="Select GONet File", command=select_file)
button.grid(row=0, column=0, padx=10, pady=10)
file_entry = Entry(root, width=50)
file_entry.grid(row=0, column=1, padx=10, pady=10)
submit_button = Button(root, text="Run Calibration", command=run_calibration_from_gui)
submit_button.grid(row=1, column=0, columnspan=2, pady=10)
root.mainloop()