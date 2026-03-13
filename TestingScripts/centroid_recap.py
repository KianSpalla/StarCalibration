#!/usr/bin/env python3
"""
Exercise: Peak finding and centroid measurement for star blobs.

Goal of this script
-------------------
You will start from labeled components (`labels`, `num_labels`) and
compute TWO positions for each star:

1. Peak pixel  = brightest pixel in the blob
2. Centroid    = flux-weighted center of brightness

This script contains step-by-step TODOs. Fill them in as you follow the
instructions during the lesson.
"""

# === Imports ===
# TODO: import numpy and matplotlib.pyplot
# import numpy as np
# import matplotlib.pyplot as plt
# Import NumPy for array operations and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from GONet_Wizard.GONet_utils import GONetFile
from pathlib import Path

# Load a GONet TIFF image file
image = Path(r"Testing Images\202_250628_063009_1751092241.jpg")
go = GONetFile.from_file(image)  # Load the file into a GONet object
go.remove_overscan()  # Remove the overscan region (non-imaging pixels)
img = go.green  # Extract the green channel as a 2D NumPy array
meta = go.meta
# print(meta)

# Enable interactive plotting mode (plots update without blocking code execution)
#plt.ion()

# Analyze the full image instead of a cutout
sub = img

# Calculate statistics for the sub-image
sub_mean = sub.mean()  # Mean pixel value (background level)
sub_std = sub.std()    # Standard deviation (noise level)

# Create a threshold to identify bright pixels (stars)
N = 2  # Threshold multiplier (higher = only brightest pixels)
threshold = sub_mean + N * sub_std  # Threshold = mean + N * std deviation
mask = sub > threshold  # Boolean array: True for pixels brighter than threshold
# Notes:
# - Full-frame mode: `sub` is the complete image.
# - Mean/std summarize the background and noise; threshold selects bright pixels.
# - `mask` marks candidate star pixels (True) based on the brightness rule.
# Define the 4 neighboring directions (up, down, left, right)
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def findClusters(mask):
    """
    Find connected clusters of True pixels in a boolean mask.
    Uses flood-fill algorithm to group connected pixels into labeled clusters.
    
    Args:
        mask: 2D boolean array where True indicates pixels to cluster
        
    Returns:
        labels: 2D int array where each cluster has a unique positive integer
        current_label: Total number of clusters found
    """
    # Create an array to store cluster labels (same shape as mask, initialized to 0)
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0  # Counter for cluster IDs (will increment for each new cluster)
    rows, cols = mask.shape
    
    # Scan through every pixel in the mask
    for r in range(rows):
        for c in range(cols):
            # Skip pixels that are not part of the mask (False/0 pixels)
            if mask[r, c] == 0:
                continue
            
            # Skip pixels that already belong to a cluster
            if labels[r, c] != 0:
                continue
            
            # Found an unlabeled True pixel - start a new cluster!
            if mask[r, c] == 1:
                # Increment cluster ID for this new cluster
                current_label += 1
                
                # Create a stack to track pixels we need to explore (flood-fill algorithm)
                comptObj = [(r, c)]
                
                # Label the starting pixel with the current cluster ID
                labels[r, c] = current_label
                
                # Explore all connected pixels using flood-fill
                while len(comptObj) > 0:
                    # Pop a pixel from the stack to examine
                    current_element = comptObj.pop()
                    
                    # Check all 4 neighbors (up, down, left, right)
                    for n in neighbors:
                        # Calculate neighbor's row and column coordinates
                        nr = current_element[0] + n[0]  # neighbor row
                        nc = current_element[1] + n[1]  # neighbor column
                        
                        # Skip neighbors that are out of bounds
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                        
                        # If neighbor is True in mask AND not yet labeled, add it to this cluster
                        if mask[nr, nc] == 1 and labels[nr, nc] == 0:
                            # Label this neighbor with the same cluster ID
                            labels[nr, nc] = current_label
                            # Add it to the stack so we can explore its neighbors too
                            comptObj.append((nr, nc))
    
    # Return the labeled array and the total number of clusters found
    return labels, current_label

# Run the cluster finding algorithm on the mask
# labels: 2D array where each cluster has a unique integer ID (1, 2, 3, ...)
# num_clusters: Total number of star clusters found
labels, num_clusters = findClusters(mask)
# Outputs:
# - `labels`: same shape as `mask`; 0 = background, 1..num_clusters = component IDs
# - `num_clusters`: number of connected bright blobs found
# ---------------------------------------------------------------------
# PART 0 — Starting point (review)
# ---------------------------------------------------------------------
# You should already have the following from your previous scripts:
#
#   sub        : the cutout image (2D array)
#   mask       : boolean array (True = bright pixel)
#   labels     : 2D int array from label_connected_components(mask)
#   num_labels : number of connected components
#
# For now, just make sure these exist (you can copy them from your
# previous exercise script). You can uncomment these prints to check:
#
# print("sub shape:", sub.shape)
# print("mask shape:", mask.shape)
# print("labels shape:", labels.shape)
# print("num_labels:", num_clusters)

"""
SANITY CHECKS
- Shapes should match: `sub.shape == mask.shape == labels.shape`.
- `num_labels` (aka `num_clusters`) is the number of detected blobs.
These prints verify that the data structures align before measurements.
"""



# ---------------------------------------------------------------------
# PART 1 — Tiny example: argmax
# ---------------------------------------------------------------------
# Before working with real image data, we will practice using np.argmax
# which tells us the INDEX of the maximum value in an array.
#
# TODO:
# 1. Create a small 1D array, for example:
#       values = np.array([3, 10, 5, 7])
# 2. Print the array.
# 3. Use idx = np.argmax(values) to find the index of the biggest value.
# 4. Print idx and the corresponding value values[idx].
#
# values = ...
# idx = ...
# print(...)
# print(...)
# Create a simple 1D array to demonstrate np.argmax
values = np.array([3, 10, 5, 7])
# print("values:", values)

# np.argmax() returns the INDEX (position) of the maximum value
# In this case, 10 is at index 1, so idx = 1
idx = np.argmax(values)
# print("Index of max value:", idx)
# print("Max value:", values[idx])

"""
ABOUT np.argmax
`np.argmax(array)` returns the index of the largest value in `array`.
We use this pattern later to identify the brightest pixel within each star blob.
"""


# ---------------------------------------------------------------------
# PART 2 — Review: np.where for getting pixels of a component
# ---------------------------------------------------------------------
# For any component ID (label_id), np.where(labels == label_id)
# returns ALL the pixel coordinates belonging to the blob.
#
# TODO:
# 1. Pick a component ID, for example label_id = 1.
# 2. Run:
#       ys, xs = np.where(labels == label_id)
# 3. Print how many pixels are in that component:
#       print(len(xs))
# 4. Also print the brightness of those pixels using:
#       fluxes = sub[ys, xs]
#       print(fluxes)
#
# label_id = 1
# ys, xs = ...
# fluxes = ...
# print(...)

# Pick the first cluster to examine
label_id = 1

# np.where() returns the coordinates of all pixels matching a condition
# Here: find all pixels in the labels array that equal label_id (1)
# Returns: ys (row indices), xs (column indices) of all matching pixels
ys, xs = np.where(labels == label_id)

# Print the number of pixels in this cluster
# print(len(xs))

# Extract the brightness values of these pixels from the sub-image
# Fancy indexing: sub[ys, xs] gets pixel values at coordinates (ys[i], xs[i])
fluxes = sub[ys, xs]
# print(fluxes)

"""
USING np.where AND FANCY INDEXING
- `np.where(labels == label_id)` returns two arrays: `ys` (rows) and `xs` (cols)
    for all pixels belonging to the chosen component.
- `sub[ys, xs]` fetches brightness values at those coordinates.
Result: `fluxes` contains all pixel values for one blob.
"""


# ---------------------------------------------------------------------
# PART 3 — Peak finding for EACH component
# ---------------------------------------------------------------------
# We will now loop over all components (1 to num_labels) and for each:
# - get ys, xs from np.where
# - compute fluxes = sub[ys, xs]
# - find max_index = np.argmax(fluxes)
# - determine the (x_peak, y_peak) of the brightest pixel
#
# TODO:
# 1. Create an empty list all_peaks = [].
# 2. Loop over each label_id from 1 to num_labels.
# 3. Inside the loop:
#       ys, xs = np.where(labels == label_id)
#       if len(xs) == 0: continue
#       fluxes = sub[ys, xs]
#       max_index = np.argmax(fluxes)
#       x_peak = xs[max_index]
#       y_peak = ys[max_index]
#       append (x_peak, y_peak) to all_peaks
#       print something like:
#            Component k: peak at (x=?, y=?)
#
# all_peaks = []
# for label_id in range(1, num_labels + 1):
#     ...
#
# print("All peaks:", all_peaks)
# Create an empty list to store peak positions for all clusters
all_peaks = []

# Loop through each cluster ID (1 to num_clusters)
for label_id in range(1, num_clusters + 1):
    # Get all pixel coordinates belonging to this cluster
    ys, xs = np.where(labels == label_id)
    
    # Skip empty clusters (safety check)
    if len(xs) == 0: continue
    
    # Get the brightness values for all pixels in this cluster
    fluxes = sub[ys, xs]
    
    # Find the index of the brightest pixel within this cluster
    # np.argmax() returns the position in the fluxes array
    max_index = np.argmax(fluxes)
    
    # Get the actual x,y coordinates of the brightest pixel
    x_peak = xs[max_index]  # Column of the brightest pixel
    y_peak = ys[max_index]  # Row of the brightest pixel
    
    # Store this peak position
    all_peaks.append((x_peak, y_peak))
    # print(f"Component {label_id}: peak at (x={x_peak}, y={y_peak})")

# print("All peaks:", all_peaks)

"""
PEAK SUMMARY
`all_peaks` holds one (x, y) per blob: the brightest pixel location.
Peaks are easy to compute but can be affected by hot/saturated or noisy pixels.
Centroids provide a more robust position by averaging weighted by brightness.
"""


# ---------------------------------------------------------------------
# PART 4 — Visualize peak locations on the image
# ---------------------------------------------------------------------
# TODO:
# 1. Extract x_peaks and y_peaks from all_peaks.
# 2. Plot the image using plt.imshow(sub, origin="lower").
# 3. Overlay the peak positions using plt.scatter.
# 4. Add labels and a title.
#
# x_peaks = ...
# y_peaks = ...
#
# plt.figure()
# plt.imshow(sub, origin="lower")
# plt.scatter(x_peaks, y_peaks,
#             s=40, edgecolors="cyan", facecolors="none",
#             label="peak pixels")
# plt.legend()
# plt.title("Peak positions for each component")
# plt.colorbar()
# plt.show()
# Extract x and y coordinates from all_peaks list
# List comprehension: iterate through all_peaks and extract x (index 0) and y (index 1)

x_peaks = []
y_peaks = []
for peak in all_peaks:
    x_peaks.append(peak[0])
    y_peaks.append(peak[1])

# # plt.figure()
# # plt.imshow(sub, origin="lower")
# # plt.scatter(x_peaks, y_peaks,
# #             s=40, edgecolors="cyan", facecolors="none",
# #             label="peak pixels")
# # plt.legend()
# # plt.title("Peak positions for each component")
# # plt.colorbar()
# # plt.show()

"""
PEAK VS CENTROID PLOT
- Peak (cyan): brightest pixel in each blob.
- Centroid (red): brightness-weighted average position (center of mass).
Centroids are typically near the geometric center and are more robust to outliers.
"""

"""
VISUALIZING PEAKS
- `origin="lower"` places (0,0) at the bottom-left (astronomy convention).
- Cyan hollow circles mark brightest-pixel positions per component.
Adjust `vmin/vmax` to improve contrast for faint stars.
"""

# ---------------------------------------------------------------------
# PART 5 — Tiny centroid example (1D weighted average)
# ---------------------------------------------------------------------
# A centroid is a weighted average where brighter pixels pull more.
#
# Example:
#   positions = [0, 1, 2]
#   fluxes    = [1, 2, 3]
#
# Centroid x = (0*1 + 1*2 + 2*3) / (1+2+3) = 8/6 = 1.33
#
# TODO:
# 1. Make x_positions = np.array([0,1,2])
# 2. Make fluxes = np.array([1,2,3])
# 3. Compute total_flux = fluxes.sum()
# 4. Compute weighted_sum = (x_positions * fluxes).sum()
# 5. Compute x_centroid = weighted_sum / total_flux
# 6. Print the results.
#
# x_positions = ...
# fluxes = ...
# total_flux = ...
# weighted_sum = ...
# x_centroid = ...
# print(...)
x_positions = np.array([0, 1, 2])
fluxes = np.array([1, 2, 3])
total_flux = fluxes.sum()
weighted_sum = (x_positions * fluxes).sum()
x_centroid = weighted_sum / total_flux
# print("Centroid x position:", x_centroid)


# ---------------------------------------------------------------------
# PART 6 — Centroid for each component (2D)
# ---------------------------------------------------------------------
# Centroid formula in 2D:
#
#   x_centroid = sum( x_i * f_i ) / sum(f_i)
#   y_centroid = sum( y_i * f_i ) / sum(f_i)
#
# TODO:
# 1. Create an empty list sources = [].
# 2. Loop over each component (1 to num_labels).
# 3. In each loop:
#       ys, xs = np.where(labels == label_id)
#       if len(xs) == 0: continue
#       fluxes = sub[ys, xs]
#
#       # Peak:
#       max_index = np.argmax(fluxes)
#       x_peak = xs[max_index]
#       y_peak = ys[max_index]
#
#       # Centroid:
#       total_flux = fluxes.sum()
#       x_cen = (xs * fluxes).sum() / total_flux
#       y_cen = (ys * fluxes).sum() / total_flux
#
#       Append a dictionary to `sources` with:
#         "label", "x_peak", "y_peak",
#         "x_centroid", "y_centroid",
#         "flux", "npix"
#
# 4. Print summary of each source.
#
# sources = []
# for label_id in range(1, num_labels + 1):
#     ...
#
# print("Measured", len(sources), "sources")
sources = []
for label_id in range(1, num_clusters + 1):
    ys, xs = np.where(labels == label_id)
    if len(xs) == 0: continue
    fluxes = sub[ys, xs]

    # Peak:
    max_index = np.argmax(fluxes)
    x_peak = xs[max_index]
    y_peak = ys[max_index]

    # Centroid:
    total_flux = fluxes.sum()
    # Compute flux-weighted centroid (center of mass)
    x_cen = (xs * fluxes).sum() / total_flux
    y_cen = (ys * fluxes).sum() / total_flux

    # Notes:
    # - Centroid formula (2D): x_cen = sum(x_i * f_i)/sum(f_i), y_cen = sum(y_i * f_i)/sum(f_i)
    # - `xs`, `ys` are coordinates; `fluxes` are brightness values.
    # - Ensure `xs`, `ys`, and `fluxes` originate from the same blob so their lengths match.
    source = {
        "label": label_id,
        "x_peak": x_peak,
        "y_peak": y_peak,
        "x_centroid": x_cen,
        "y_centroid": y_cen,
        "flux": total_flux,
        "npix": len(xs)
    }
    sources.append(source)
    # print(f"Source {label_id}: peak=({x_peak},{y_peak}), "
    #       f"centroid=({x_cen:.2f},{y_cen:.2f}), "
    #       f"flux={total_flux}, npix={len(xs)}")

# ---------------------------------------------------------------------
# PART 7 — Visualize peak vs centroid
# ---------------------------------------------------------------------
# TODO:
# 1. Extract lists x_peak_list, y_peak_list, x_cen_list, y_cen_list.
# 2. Plot the image (plt.imshow).
# 3. Overlay peaks (cyan) and centroids (red).
#
# x_peak_list = ...
# y_peak_list = ...
# x_cen_list = ...
# y_cen_list = ...
#
# plt.figure()
# plt.imshow(sub, origin="lower")
#
# plt.scatter(x_peak_list, y_peak_list,
#             s=30, edgecolors="cyan", facecolors="none",
#             label="Peak")
#
# plt.scatter(x_cen_list, y_cen_list,
#             s=50, edgecolors="red", facecolors="none",
#             label="Centroid")
#
# plt.legend()
# plt.title("Peak vs centroid positions")
# plt.colorbar()
# plt.show()

y_peak_list = []
x_peak_list = []
y_cen_list = []
x_cen_list = []
for source in sources:
    x_peak_list.append(source.get("x_peak"))
    y_peak_list.append(source.get("y_peak"))
    x_cen_list.append(source.get("x_centroid"))
    y_cen_list.append(source.get("y_centroid"))

plt.figure()
plt.imshow(sub, origin="lower")

plt.scatter(x_peak_list, y_peak_list,
            s=30, edgecolors="cyan", facecolors="none",
            label="Peak")

plt.scatter(x_cen_list, y_cen_list,
            s=50, edgecolors="red", facecolors="none",
            label="Centroid")
plt.legend()
plt.title("Peak vs centroid positions")
plt.colorbar()
plt.show()


# ---------------------------------------------------------------------
# OPTIONAL PART 8 — Wrap into a function
# ---------------------------------------------------------------------
# TODO (optional):
# Write a function measure_sources(sub, labels, num_labels)
# that returns the `sources` list.
#
# def measure_sources(sub, labels, num_labels):
#     """
#     Measure peak and centroid for each labeled component.
#     """
#     ...
#     return sources

def measure_sources(sub, labels, num_clusters):
    sources = []
    x_centroids = []
    y_centroids = []
    for label_id in range(1, num_clusters + 1):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0: continue
        fluxes = sub[ys, xs]
        
        # Peak:
        max_index = np.argmax(fluxes)
        x_peak = xs[max_index]
        y_peak = ys[max_index]

        # Centroid:
        total_flux = fluxes.sum()
        x_centroid = (xs * fluxes).sum() / total_flux
        y_centroid = (ys * fluxes).sum() / total_flux

        source = {
            "label": label_id,
            "x_peak": x_peak,
            "y_peak": y_peak,
            "x_centroid": x_centroid,
            "y_centroid": y_centroid,
            "flux": total_flux,
            "npix": len(xs)
        }
        sources.append(source)
        x_centroids.append(x_centroid)
        y_centroids.append(y_centroid)
    return sources, x_centroids, y_centroids

# print(sources)



# End of script.
