from operator import sub

import numpy as np
from scipy.ndimage import shift as nd_shift
from PIL import Image
from geometry import predict_pixels_from_catalog


def find_zenith_pixel_and_center(img, best, cx, cy, R_pix):
    zenith_pred = predict_pixels_from_catalog(np.array([90.0]), np.array([0.0]),cx, cy, R_pix, best["alpha"], best["beta"], best["gamma"],)
    zenith_x = float(zenith_pred[0, 0])
    zenith_y = float(zenith_pred[0, 1])

    target_cx = (img.shape[1] - 1) / 2.0
    target_cy = (img.shape[0] - 1) / 2.0
    shift_x = target_cx - zenith_x
    shift_y = target_cy - zenith_y

    centered_sub = nd_shift(img.astype(float), shift=(shift_y, shift_x), order=1, mode="constant", cval=float(np.median(img)),)

    return { 
        "zenith_x": float(zenith_x), 
        "zenith_y": float(zenith_y), 
        "target_cx": float(target_cx), 
        "target_cy": float(target_cy), 
        "shift_x": float(shift_x), 
        "shift_y": float(shift_y),
        "centered_sub": centered_sub,
    }


def build_shifted_image(image_path, shift_x, shift_y):
    if not image_path or shift_x is None or shift_y is None:
        raise ValueError("image_path, shift_x, and shift_y must all be provided.")

    image_array = np.array(Image.open(image_path))
    shift = (float(shift_y), float(shift_x)) if image_array.ndim == 2 else (float(shift_y), float(shift_x), 0)
    shifted = nd_shift(image_array.astype(float), shift=shift, order=1, mode="constant", cval=float(np.median(image_array)))
    shifted = np.clip(shifted, 0, 255).astype(image_array.dtype)
    return Image.fromarray(shifted)
