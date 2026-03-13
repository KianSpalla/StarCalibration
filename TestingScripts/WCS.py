#!/usr/bin/env python3
"""
GUIDED EXERCISE: From centroids to a WCS solution (Astropy) — with an altitude cut

New info you provided
---------------------
- Image center (cx, cy) = (1030, 760)
- Image radius R_pix    = 760 pixels corresponds to 90 degrees from zenith (horizon)

This means:
- A point at the horizon is ~90° from the center in angular distance from zenith.
- If we want to keep only stars within 80° of zenith (i.e., not too close to horizon),
  we can filter by pixel distance from the center.

Key idea
--------
If R_pix maps to 90°, then:
    degrees_from_zenith  ~  (pixel_distance / R_pix) * 90

So to keep only <= deg_thresh:
    pixel_distance <= (deg_thresh / 90) * R_pix

Today you will:
- Write a small function that filters points by an angular threshold (degrees)
- Use it to reduce both:
    (a) image centroids
    (b) transformed catalog pixel points
- Then proceed with matching and (optionally) WCS fitting

This script is a guide: it gives the tools and the plan, but you fill the TODOs.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from GONet_Wizard.GONet_utils import GONetFile
from centroid_recap import measure_sources
from GoNet_Exercise_DecemberSixth import compact_search
from astrometry_1 import stars_icrs, x, y
from pathlib import Path

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# =============================================================================
# GIVEN CONSTANTS (from instructor)
# =============================================================================
CX, CY = 1030.0, 760.0
R_PIX = 760.0  # 90 degrees


# =============================================================================
# STEP 0 — Load / define your inputs
# =============================================================================
# TODO:
# Provide:
# - img_gray : 2D grayscale image
# - sources  : list of dicts with x_centroid, y_centroid, flux
# - cat_sky and cat_xy_unit from your catalog query/projection script
#
image_path = Path(r"Testing Images\256_251029_204008_1761770474.jpg")
go = GONetFile.from_file(image_path)
N=5
go_ravel=np.ravel(go.green)
img_gray = go.green
go.remove_overscan()
mean = np.mean(go_ravel)
stddev = np.std(go_ravel)
threshold = mean + N * stddev
mask = img_gray > threshold
labels, num_clusters = compact_search(np.array(mask))      # TODO
sources, x_centroids, y_centroids = measure_sources(img_gray, labels, num_clusters)       # TODO
cat_sky = stars_icrs           # TODO  (Astropy SkyCoord)
cat_xy_unit = np.column_stack((x, y))      # TODO  (M,2) unit disk coords aligned with cat_sky


# =============================================================================
# STEP 1 — Build img_xy and select bright stars
# =============================================================================
# TODO:
# 1) Sort by flux descending
# 2) Pick K brightest
# 3) Build img_xy array shape (K,2)
#
sources_sorted = sorted(sources, key=lambda d: d["flux"], reverse=True)
K = 300
img_xy = np.array([[d["x_centroid"], d["y_centroid"]] for d in sources_sorted[:K]], dtype=float)


# =============================================================================
# STEP 2 — FILTERING BY DISTANCE-TO-CENTER (deg threshold)
# =============================================================================
# We want a function that filters points in pixel coordinates (x,y) by:
#   distance from (cx,cy) converted into degrees from zenith.
#
# Reminder:
#   R_pix corresponds to 90 degrees
# so:
#   deg = (dist_pix / R_pix) * 90
#
# TODO:
# Implement the function below.
#
def filter_by_zenith_angle(
    xy_pix: np.ndarray,
    cx: float,
    cy: float,
    r_pix_90deg: float,
    deg_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter points by their angular distance from the zenith (center).

    Parameters
    ----------
    xy_pix : (N,2) np.ndarray
        Points in pixel coordinates (x,y).
    cx, cy : float
        Image center (zenith) in pixels.
    r_pix_90deg : float
        Pixel radius corresponding to 90 degrees from zenith (horizon).
    deg_thresh : float
        Keep only points with angle <= deg_thresh.

    Returns
    -------
    xy_keep : (K,2) np.ndarray
        Filtered points that satisfy the threshold.
    keep_mask : (N,) np.ndarray (bool)
        Boolean mask for which points were kept.

    Guidance / Hints
    ----------------
    1) Compute dx = x - cx and dy = y - cy
    2) Compute dist_pix = sqrt(dx^2 + dy^2)
    3) Convert to degrees: deg = (dist_pix / r_pix_90deg) * 90
    4) keep_mask = deg <= deg_thresh
    5) return xy_pix[keep_mask], keep_mask
    """
    # TODO: implement
    dx = xy_pix[:, 0] - cx
    dy = xy_pix[:, 1] - cy
    dist_pix = np.sqrt(dx**2 + dy**2)
    deg = (dist_pix / r_pix_90deg) * 90.0
    keep_mask = deg <= deg_thresh
    return xy_pix[keep_mask], keep_mask
    #raise NotImplementedError("TODO: implement filter_by_zenith_angle")

# TODO:
# Use your function to filter image centroids to within 80 degrees.
#
# deg_thresh = 80.0
# img_xy_filt, img_keep = filter_by_zenith_angle(img_xy, CX, CY, R_PIX, deg_thresh)
img_xy_filt, img_keep = filter_by_zenith_angle(img_xy, CX, CY, R_PIX, 80.0)
print(f"Filtered image centroids: {img_xy.shape[0]} -> {img_xy_filt.shape[0]} points within 80°")

# TODO (recommended):
# Plot before/after filtering for image centroids
plt.figure(figsize=(8, 8))
plt.imshow(img_gray, origin="lower")
plt.scatter(img_xy[:,0], img_xy[:,1], s=10, edgecolors="yellow", facecolors="none", label="all (bright subset)")
plt.scatter(img_xy_filt[:,0], img_xy_filt[:,1], s=15, edgecolors="lime", facecolors="none", label="<= 80 deg")
plt.legend()
plt.title("Image centroids: filtering by zenith angle")
plt.axis("off")
plt.show()


# =============================================================================
# STEP 3 — Prepare bright catalog subset (aligned with cat_sky)
# =============================================================================
# TODO:
# Decide how to select a manageable catalog subset.
#
# You can do either:
# - limit by magnitude (if you have gmag array)
# - limit by count (first M points, or sort then slice)
#
# IMPORTANT:
# Keep cat_sky and cat_xy_unit aligned.
#
M = 300  # number of brightest catalog stars to keep
cat_sky_bright = cat_sky[:M]
cat_xy_unit_bright = cat_xy_unit[:M]
print(f"Selected {M} brightest catalog stars")


# =============================================================================
# STEP 4 — Transform catalog unit coords into pixels (rotation + flips)
# =============================================================================
def rot2d(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


def transform_catalog_to_pixels(
    cat_xy_unit: np.ndarray,
    cx: float, cy: float, r_pix: float,
    theta_deg: float,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    xy = cat_xy_unit.copy()
    if flip_x:
        xy[:, 0] *= -1
    if flip_y:
        xy[:, 1] *= -1
    R = rot2d(np.deg2rad(theta_deg))
    xy = xy @ R.T
    out = np.empty_like(xy)
    out[:, 0] = cx + r_pix * xy[:, 0]
    out[:, 1] = cy + r_pix * xy[:, 1]
    return out

# =============================================================================
# STEP 5 — Nearest neighbor tool (catalog -> image)
# =============================================================================
def nearest_neighbor(img_xy: np.ndarray, query_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find nearest image point for each query point.

    Returns
    -------
    dists : (M,) distances to nearest image centroid
    idx   : (M,) indices into img_xy
    """
    if HAVE_SCIPY:
        tree = cKDTree(img_xy)
        dists, idx = tree.query(query_xy, k=1)
        return dists, idx

    dists = np.empty(query_xy.shape[0], dtype=float)
    idx = np.empty(query_xy.shape[0], dtype=int)
    for i, p in enumerate(query_xy):
        dd = np.sqrt(((img_xy - p) ** 2).sum(axis=1))
        j = int(np.argmin(dd))
        dists[i] = dd[j]
        idx[i] = j
    return dists, idx


# =============================================================================
# STEP 6 — Score alignment + search rotation/flip
# =============================================================================
def score_alignment(img_xy: np.ndarray, cat_xy_pix: np.ndarray, tol_pix: float) -> int:
    """
    Score = number of catalog points within tol_pix of some image centroid.

    TODO:
    - use nearest_neighbor()
    - count distances <= tol_pix
    """
    # TODO: implement
    dists, idx = nearest_neighbor(img_xy, cat_xy_pix)
    count = np.sum(dists <= tol_pix)
    return count
    raise NotImplementedError("TODO: implement score_alignment")


# TODO:
# Write the search loop:
# - pick tol_pix (coarse matching tolerance) e.g. 25–40 px
# - pick step_deg (coarse rotation step) e.g. 3–5 degrees
# - try flip_x/flip_y booleans
# - for each trial:
#     cat_pix = transform_catalog_to_pixels(...)
#     OPTIONAL: filter catalog pixels by zenith angle too (<= 80°)
#     s = score_alignment(img_xy_filt, cat_pix_filt, tol_pix)
# - keep best
#
tol_pix = 25.0
step_deg = 3.0
flip_x = False
flip_y = False
best_score = -1
for flip_x in [False, True]:
    for flip_y in [False, True]:
        for theta_deg in np.arange(0, 360, step_deg):
            cat_pix = transform_catalog_to_pixels(
                cat_xy_unit_bright,
                CX, CY, R_PIX,
                theta_deg,
                flip_x=flip_x,
                flip_y=flip_y,
                
            )
            cat_pix_filt, cat_keep = filter_by_zenith_angle(cat_pix, CX, CY, R_PIX, 80.0)
            s = score_alignment(img_xy_filt, cat_pix_filt, tol_pix)
            if s > best_score:
                best_score = s
                best = {
                    "theta_deg": theta_deg,
                    "flip_x": flip_x,
                    "flip_y": flip_y,
                    "score": s,
                }
            #print(f"theta={theta_deg:.1f} flip_x={flip_x} flip_y={flip_y} score={s}") 

  # TODO best dict
print("Best alignment:")
print(f"  rotation (deg): {best['theta_deg']}")
print(f"  flip_x:        {best['flip_x']}")
print(f"  flip_y:        {best['flip_y']}")
print(f"  score:         {best['score']}")


# =============================================================================
# STEP 7 — Build matched pairs (pixel <-> sky)
# =============================================================================
# Now that you have a best transform, you can create real matches:
#
# 1) Transform the catalog (bright subset) to pixel coords using best params
# 2) Filter catalog points to <= 80° like you did for image points
# 3) Find nearest image centroid for each catalog point
# 4) Keep pairs with distance < tol_match (e.g. 15–25 px)
# 5) Deduplicate: each image centroid can be used at most once
#
# TODO:
# Implement this carefully and print the number of matches.
#
cat_pix_best = transform_catalog_to_pixels(
    cat_xy_unit_bright,
    CX, CY, R_PIX,
    best['theta_deg'],
    flip_x=best['flip_x'],
    flip_y=best['flip_y'],
)
cat_pix_best_filt, cat_keep = filter_by_zenith_angle(cat_pix_best, CX, CY, R_PIX, 80.0)
dists, idx = nearest_neighbor(img_xy_filt, cat_pix_best_filt)
tol_match = 20.0
matched_img_indices = []
matched_cat_indices = []
for cat_i, (dist, img_i) in enumerate(zip(dists, idx)):
    if dist < tol_match:
        if img_i not in matched_img_indices:
            matched_img_indices.append(img_i)
            matched_cat_indices.append(cat_i)
print(f"Number of matched pairs: {len(matched_img_indices)}")

matched_img_xy = img_xy_filt[matched_img_indices]
matched_cat_sky = cat_xy_unit_bright[cat_keep][matched_cat_indices]



# =============================================================================
# STEP 8 (OPTIONAL EXTRA) — Fit WCS with Astropy
# =============================================================================
# If you have >= 30 good matches, try a TAN WCS fit.
#
# Astropy tool:
#   from astropy.wcs.utils import fit_wcs_from_points
#
# What it does:
# - Fits a WCS that best maps pixel points to provided RA/Dec points.
#
# TODO:
# 1) Import fit_wcs_from_points
# 2) Build x,y arrays from matched_img_xy
# 3) Fit:
#       wcs = fit_wcs_from_points((x, y), matched_cat_sky, projection="TAN")
# 4) Print wcs
#
wcs = None  # TODO


# =============================================================================
# STEP 9 (OPTIONAL EXTRA) — Validate WCS with residuals
# =============================================================================
# Use:
#   x_pred, y_pred = wcs.world_to_pixel(matched_cat_sky)
# Compare against matched_img_xy.
#
# TODO:
# - compute residuals
# - print median / 90% / max
# - overlay x_pred/y_pred vs matched_img_xy on the image
#
# Expected behavior:
# - if matches are correct: residuals should be "reasonable"
# - fisheye distortions make residuals larger near edges, which is why we cut to 80°
#
# If residuals huge:
# - check matching tolerances
# - reduce to brighter stars
# - ensure cx, cy, R_pix correct
# - ensure you filtered consistently


# =============================================================================
# MAIN
# =============================================================================
def main():
    # This is a guided script; you will uncomment/run step-by-step during tutoring.
    # Start by implementing filter_by_zenith_angle and testing it with img_xy.
    pass


if __name__ == "__main__":
    main()