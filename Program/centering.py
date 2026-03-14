import numpy as np
from scipy.ndimage import shift as nd_shift, rotate as nd_rotate
from PIL import Image
from geometry import predict_pixels_from_catalog

"""
find_zenith_pixel_and_center takes an image, the best orientation parameters, the center coordinates, and the radius in pixels.
calls the predict_pixels_from_catalog function to find the predicted pixels of the zenith based on the best orientation parameters. 
It then calculates the shift needed to move the zenith pixel to the center of the image and applies that shift,
then rotates the image by -alpha around the center to correct for the camera's azimuthal rotation.
"""
def find_zenith_pixel_and_center(img, best, cx, cy, radiusPix):
    zenithPrediction = predict_pixels_from_catalog(np.array([90.0]), np.array([0.0]),cx, cy, radiusPix, best["alpha"], best["beta"], best["gamma"],)
    zenithX = float(zenithPrediction[0, 0])
    zenithY = float(zenithPrediction[0, 1])

    targetCenterX = (img.shape[1] - 1) / 2.0
    targetCenterY = (img.shape[0] - 1) / 2.0
    shiftX = targetCenterX - zenithX
    shiftY = targetCenterY - zenithY

    alphaDeg = float(np.rad2deg(best["alpha"]))

    centeredSub = nd_shift(img.astype(float), shift=(shiftY, shiftX), order=1, mode="constant", cval=float(np.median(img)),)
    centeredSub = nd_rotate(centeredSub, -alphaDeg, reshape=False, order=1, mode="constant", cval=float(np.median(img)),)

    return { 
        "zenithX": float(zenithX), 
        "zenithY": float(zenithY), 
        "targetCenterX": float(targetCenterX), 
        "targetCenterY": float(targetCenterY), 
        "shiftX": float(shiftX), 
        "shiftY": float(shiftY),
        "alphaDeg": alphaDeg,
        "centeredSub": centeredSub,
    }


"""
build_shifted_image takes an image path, the shift values in x and y directions, and the alpha rotation in degrees.
It loads the image, shifts it to center the zenith, then rotates by -alphaDeg around the center
and returns the corrected image as a PIL Image object.
"""
def build_shifted_image(imagePath, shiftX, shiftY, alphaDeg):
    if not imagePath or shiftX is None or shiftY is None:
        raise ValueError("imagePath, shiftX, and shiftY must all be provided.")

    imageArray = np.array(Image.open(imagePath))
    shift = (float(shiftY), float(shiftX)) if imageArray.ndim == 2 else (float(shiftY), float(shiftX), 0)
    shifted = nd_shift(imageArray.astype(float), shift=shift, order=1, mode="constant", cval=float(np.median(imageArray)))
    shifted = nd_rotate(shifted, -float(alphaDeg), reshape=False, order=1, mode="constant", cval=float(np.median(imageArray)))
    shifted = np.clip(shifted, 0, 255).astype(imageArray.dtype)
    return Image.fromarray(shifted)
