import numpy as np
from scipy.ndimage import shift as nd_shift
from PIL import Image
from geometry import predict_pixels_from_catalog


def find_zenith_pixel_and_center(sub, best, cx, cy, R_pix):
    """
    Projects the zenith (alt=90°, any az) through the fisheye model to find
    its pixel location, then shifts the image so the zenith lands at centre.
    """
    zenith_pred = predict_pixels_from_catalog(
        np.array([90.0]), np.array([0.0]),
        cx, cy, R_pix,
        best["alpha"], best["beta"], best["gamma"],
    )
    zenith_x = float(zenith_pred[0, 0])
    zenith_y = float(zenith_pred[0, 1])

    target_cx = (sub.shape[1] - 1) / 2.0
    target_cy = (sub.shape[0] - 1) / 2.0
    shift_x = target_cx - zenith_x
    shift_y = target_cy - zenith_y

    centered_sub = nd_shift(
        sub.astype(float),
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=float(np.median(sub)),
    )

    return {
        "zenith_x": float(zenith_x),
        "zenith_y": float(zenith_y),
        "target_cx": float(target_cx),
        "target_cy": float(target_cy),
        "shift_x": float(shift_x),
        "shift_y": float(shift_y),
        "centered_sub": centered_sub,
    }


def build_shifted_image_same_format(image_path, shift_x, shift_y):
    pil_image = Image.open(image_path)
    original_mode = pil_image.mode
    original_format = pil_image.format or "PNG"
    original_suffix = str(image_path).rsplit(".", 1)[-1].lower() if "." in str(image_path) else "png"

    image_array = np.array(pil_image)
    original_dtype = image_array.dtype
    shift_kwargs = dict(order=1, mode="constant")

    if image_array.ndim == 2:
        cval = float(np.median(image_array))
        shifted = nd_shift(
            image_array.astype(float),
            shift=(float(shift_y), float(shift_x)),
            cval=cval,
            **shift_kwargs,
        )
    elif image_array.ndim == 3:
        shifted_channels = []
        for ch in range(image_array.shape[2]):
            channel = image_array[..., ch]
            cval = float(np.median(channel))
            shifted_channels.append(
                nd_shift(
                    channel.astype(float),
                    shift=(float(shift_y), float(shift_x)),
                    cval=cval,
                    **shift_kwargs,
                )
            )
        shifted = np.stack(shifted_channels, axis=-1)
    else:
        raise ValueError(
            f"Unsupported image array shape {image_array.shape}. "
            "Expected 2-D (greyscale) or 3-D (colour) array."
        )

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        shifted = np.clip(shifted, info.min, info.max).astype(original_dtype)
    else:
        shifted = shifted.astype(original_dtype)

    try:
        shifted_image = Image.fromarray(shifted, mode=original_mode)
    except (TypeError, ValueError):
        shifted_image = Image.fromarray(shifted)

    return {
        "shifted_image": shifted_image,
        "shifted_format": original_format,
        "suggested_suffix": f".{original_suffix}",
    }
