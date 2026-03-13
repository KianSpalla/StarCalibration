#!/usr/bin/env python3
"""
Intro exercise: finding star pixels in a GONet image.

Goal of this script
-------------------
Fill in the code step by step so that by the end you can:

1. Load a GONet image as a NumPy array.
2. Explore its pixel values (min, max, mean, std, histogram).
3. Extract a small sub-image (cutout).
4. Apply a brightness threshold to find "star pixels".
5. Visualize star pixels overlaid on the image.
"""

# === Imports ===
# TODO: import numpy, matplotlib.pyplot, and an image loading function (e.g., imageio.v3 or plt.imread)
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from GONet_Wizard.GONet_utils import GONetFile

import numpy as np
import matplotlib.pyplot as plt
from GONet_Wizard.GONet_utils import GONetFile
from pathlib import Path

# ---------------------------------------------------------------------
# EXERCISE 0.1 — Load an image
# ---------------------------------------------------------------------
# 1. Choose a GONet image file (set a Path pointing to it).
image = Path(r"C:\Users\spall\Desktop\GONet\Testing Images\256_251029_204008_1761770474.tiff")
# 2. Load the image
go = GONetFile.from_file(image)
go.remove_overscan()
# 3. Save one of the channles to a variable, e.g. `img`
img = go.green
plt.ion()
# 4. Print:
print(type(img))
print(img.shape)
print(img.dtype)
#    - type(img)
#    - img.shape
#    - img.dtype
#
# Hints:
# - Use GONetFile.from_file(path).
#
# img = ...
# print(type(img))
# print(img.shape)
# print(img.dtype)


# ---------------------------------------------------------------------
# EXERCISE 0.2 — Convert to grayscale (if needed)
# ---------------------------------------------------------------------
# Visualize the image
#
# Hints:
# - Display using plt.imshow
plt.figure()
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.show()


# ---------------------------------------------------------------------
# EXERCISE 1.1 — Summary statistics
# ---------------------------------------------------------------------
# Compute and print the following for img:
# - minimum value
# - maximum value
# - mean value
# - standard deviation
#
# Use:
#   img.min(), img.max(), img.mean(), img.std()
#

minVal = img.min()
maxVal = img.max()
meanVal = img.mean()
stdVal = img.std()

print("min:", minVal)
print("max:", maxVal)
print("mean:", meanVal)
print("std:", stdVal)

# min_val = ...
# max_val = ...
# mean_val = ...
# std_val = ...
# print("min:", min_val)
# print("max:", max_val)
# print("mean:", mean_val)
# print("std:", std_val)


# ---------------------------------------------------------------------
# EXERCISE 1.2 — Histogram of pixel values
# ---------------------------------------------------------------------
# Plot a histogram of all pixel values in img.
#
# Steps:
# - Flatten the image with img.ravel()
# - Use plt.hist(..., bins=100)
# - Label the x-axis as "Pixel value" and y-axis as "Count"

img.ravel()
plt.figure()
plt.hist(img.ravel(), bins=1000)
plt.xlabel("Pixel value")

plt.ylabel("Count")

plt.show()
#
# plt.figure()
# plt.hist(..., bins=100)
# plt.xlabel("Pixel value")
# plt.ylabel("Count")
# plt.title("Histogram of pixel values")
# plt.show()


# ---------------------------------------------------------------------
# EXERCISE 2.1 — Extract a sub-image (cutout)
# ---------------------------------------------------------------------
# Choose a rectangular region of the image that contains stars.
# For example, pick rows [y0:y1] and columns [x0:x1].
#
# Steps:
# - Define y0, y1, x0, x1.
# - Extract sub = img[y0:y1, x0:x1]
# - Print sub.shape.
# - Display sub with imshow + colorbar.
#
y0, y1 = 500, 700    # example; adjust to where stars are
x0, x1 = 500, 700
sub = img[y0:y1, x0:x1]
print("sub shape:", sub.shape)
plt.figure()
plt.imshow(sub, origin="lower")
plt.colorbar()
plt.title("Sub-image (cutout)")
plt.show()

# sub = ...
# print("sub shape:", sub.shape)
#
# plt.figure()
# plt.imshow(sub, origin="lower")
# plt.colorbar()
# plt.title("Sub-image (cutout)")
# plt.show()


# ---------------------------------------------------------------------
# EXERCISE 2.2 — Adjust contrast on the sub-image
# ---------------------------------------------------------------------
# We want to change vmin and vmax to better see stars.
#
# Steps:
# - Compute mean and std of sub: sub_mean, sub_std.
# - Display sub with:
#       vmin = sub_mean - 1 * sub_std
#       vmax = sub_mean + 5 * sub_std
# - Experiment: try changing the multipliers (e.g. 0.5, 2, 10) and see how
#   the visibility of stars changes.
#
# sub_mean = ...
# sub_std = ...
#
sub_mean = sub.mean()
sub_std = sub.std()
plt.figure()
plt.imshow(sub, origin="lower",
           vmin=sub_mean - 1 * sub_std,
           vmax=sub_mean + 5 * sub_std)
plt.colorbar()
plt.title("Sub-image with adjusted contrast")
plt.show()
# plt.figure()
# plt.imshow(sub, origin="lower",
#            vmin=sub_mean - 1 * sub_std,
#            vmax=sub_mean + 5 * sub_std)
# plt.colorbar()
# plt.title("Sub-image with adjusted contrast")
# plt.show()


# ---------------------------------------------------------------------
# EXERCISE 3.1 — Create a threshold mask for star pixels
# ---------------------------------------------------------------------
# We want to mark "bright" pixels as candidate star pixels using a simple rule:
#
#   threshold = sub_mean + N * sub_std
#
# where N is something like 5 (you can try 3, 5, 7, etc.).
#
# Steps:
# - Choose N (e.g. N = 5).
# - Compute threshold.
# - Create a boolean mask:
#       mask = sub > threshold
# - Display mask using imshow(mask, origin="lower", cmap="gray").
#
# N = 5
# threshold = ...
# mask = ...
#
# plt.figure()
# plt.imshow(mask, origin="lower", cmap="gray")
# plt.title(f"Mask of star pixels (N = {N})")
# plt.show()
N = 5
threshold = sub_mean + N * sub_std
mask = sub > threshold
plt.figure()
plt.imshow(mask, origin="lower", cmap="gray")
plt.title(f"Mask of star pixels (N = {N})")
plt.show()


# ---------------------------------------------------------------------
# EXERCISE 3.2 — Count star pixels and fraction
# ---------------------------------------------------------------------
# Now we want to know how many pixels are marked as "star pixels".
#
# Steps:
# - Count them using np.sum(mask).
# - Compute the fraction of pixels above the threshold using mask.mean().
# - Print both.
#
# num_star_pixels = ...
# fraction_star_pixels = ...
# print("Number of star pixels:", num_star_pixels)
# print("Fraction of pixels above threshold:", fraction_star_pixels)
#
# Try changing N (for example 3, 5, 7) and see how these numbers change.
num_star_pixels = np.sum(mask)
fraction_star_pixels = mask.mean()
print("Number of star pixels:", num_star_pixels)
print("Fraction of pixels above threshold:", fraction_star_pixels)


# ---------------------------------------------------------------------
# EXERCISE 4.1 — Get coordinates of star pixels
# ---------------------------------------------------------------------
# We want the (x, y) coordinates of all pixels where mask is True.
#
# Steps:
# - Use np.where(mask) → returns (ys, xs).
# - Print how many coordinates you got (should match num_star_pixels).
#
# ys, xs = np.where(mask)
# print("Number of coordinates:", len(xs))
ys, xs = np.where(mask)
print("Number of coordinates:", len(xs))


# ---------------------------------------------------------------------
# EXERCISE 4.2 — Overlay star pixels on the sub-image
# ---------------------------------------------------------------------
# Finally, let's visualize which pixels we detected as "star pixels" on top
# of the original sub-image.
#
# Steps:
# - Make a new figure.
# - Show sub with imshow (using vmin/vmax like before).
# - Use plt.scatter(xs, ys, ...) to overlay small circles at the star pixels.
#   (remember: xs = columns, ys = rows)
#
# Example:
#   plt.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')
#
# plt.figure()
# plt.imshow(sub, origin="lower",
#            vmin=sub_mean - 1 * sub_std,
#            vmax=sub_mean + 5 * sub_std)
# plt.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')
# plt.title("Detected star pixels")
# plt.colorbar()
# plt.show()
plt.figure()
plt.imshow(sub, origin="lower",
    vmin=sub_mean - 1 * sub_std,
    vmax=sub_mean + 5 * sub_std)
plt.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')
plt.title("Detected star pixels")
plt.colorbar()
plt.show()


# ---------------------------------------------------------------------
# OPTIONAL EXERCISE 5 — Experiment with different thresholds
# ---------------------------------------------------------------------
# Try running the mask + overlay steps multiple times with different N values
# (e.g. 3, 4, 5, 6, 7) and observe:
# - How does the number of star pixels change?
# - Do you start picking up too much noise when N is small?
# - Do you start missing fainter stars when N is large?
#
# You can wrap the detection and plotting into a small function, like:
#
# def show_star_pixels(sub, N):
#     """
#     For a given sub-image and threshold factor N, detect star pixels and plot.
#     """
#     # TODO: compute mean/std, threshold, mask, then plot sub + star pixels
#     pass
#
# Then call:
#   show_star_pixels(sub, 3)
#   show_star_pixels(sub, 5)
#   show_star_pixels(sub, 7)
#
# End of exercise script.
def show_star_pixels(sub, N):
    sub_mean = sub.mean()
    sub_std = sub.std()
    threshold = sub_mean + N * sub_std
    mask = sub > threshold
    ys, xs = np.where(mask)
    #findClusters(mask)

    plt.figure()
    plt.imshow(sub, origin="lower",
        vmin=sub_mean - 1 * sub_std,
        vmax=sub_mean + 5 * sub_std)
    plt.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')
    plt.title(f"Detected star pixels (N={N})")
    plt.colorbar()
    plt.show()

#show_star_pixels(sub, 3)
#show_star_pixels(sub, 5)
#show_star_pixels(sub, 7)

maskdemo = np.array([[False, False, True,  False, False],
   [False, False, True,  False, False],
   [True,  True,  True,  False, False],
   [False, False, False, False, True ],
   [False, False, False, False, True ],])


neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def findClusters(mask):
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 0:
                continue
            
            if labels[r, c] != 0:
                continue
            
            if mask[r, c] == 1:
                current_label += 1
                comptObj = [(r, c)]
                labels[r, c] = current_label
                
                while len(comptObj) > 0:
                    current_element = comptObj.pop()
                    for n in neighbors:
                        nr = current_element[0] + n[0]
                        nc = current_element[1] + n[1]
                        
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                        
                        if mask[nr, nc] == 1 and labels[nr, nc] == 0:
                            labels[nr, nc] = current_label
                            comptObj.append((nr, nc))
    return labels, current_label

# Use the actual mask from the sub-image instead of maskdemo
labels, num_clusters = findClusters(mask)

plt.imshow(sub, origin="lower", vmax=sub_mean + 5 * sub_std, vmin=sub_mean - 1 * sub_std)
plt.title("Detected star pixels with clusters")
for cluster in range(1, num_clusters + 1):
    #plt.cla()
    yc, xc = np.where(labels == cluster)
    star = sub[yc, xc]

    flux = np.sum(star)

    # peak = np.argmax(star)
    # y_peak = yc[peak]
    # x_peak = xc[peak]
    # print(y_peak, x_peak)
    # plt.plot(x_peak, y_peak, 'r.')

    #get the weighted centroid
    centroidY = np.sum(yc * star) / flux
    centroidX = np.sum(xc * star) / flux
    plt.plot(centroidX, centroidY, 'r.')

    
input()
# # Visualize on full image
# plt.figure(figsize=(12, 10))
# plt.imshow(img, origin="lower", vmin=img.mean() - img.std(), vmax=img.mean() + 3*img.std())

# for i in range(1, num_clusters + 1):
#     coords = np.argwhere(labels == i)
#     full_rows = coords[:, 0] + y0
#     full_cols = coords[:, 1] + x0
#     plt.scatter(full_cols, full_rows, s=20, label=f'Cluster {i}')
# plt.show()

