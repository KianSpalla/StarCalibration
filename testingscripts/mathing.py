#!/usr/bin/env python3
"""
GUIDED SCRIPT: A gentle first star-matching experiment (image centroids ↔ catalog stars)

You already have:
  - Image star centroids: (x, y) in pixels (from connected components + centroid code)
  - A catalog query (e.g., Gaia) that you projected into a zenith-centered fisheye plot:
      (x_cat_unit, y_cat_unit) in a unit disk (horizon radius ~ 1)

TODO:
  1) Convert catalog unit-disk coordinates into IMAGE pixel coordinates using:
        - image center (cx, cy)
        - image radius (R_pix)
        - a trial rotation angle theta
        - possible flips (mirror)
  2) Measure how good a trial transform is using a nearest-neighbor "match score"
  3) Search over theta + flips and find a "best" configuration
  4) Plot an overlay to visually confirm the result

IMPORTANT CONCEPTS
------------------
- We are NOT doing a full astrometric solve.
- This is a "toy" matching step to get the orientation (rotation/flip) approximately right.
- A fisheye lens introduces distortions; expect imperfect alignment especially near horizon.

EXTRA (optional, at the end)
----------------------------
Once you have reasonable matches, you can attempt an Astropy WCS fit using the matched pairs.
We will include a small optional section at the end with the right tools.

Dependencies
------------
- numpy, matplotlib
- scipy (optional but recommended; used for fast nearest neighbor search)

Tip
---
Start with BRIGHT stars only (small catalog magnitude, high image flux).
Matching gets much easier when you only use the top few hundred stars.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from GONet_Wizard.GONet_utils import GONetFile
from pathlib import Path
from centroid import compute_img_centroids
from astrometry import compute_cat_unit_coords

# Optional speed-up for nearest neighbors
try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# =============================================================================
# STEP 0 — Load your inputs (YOU provide these from previous scripts)
# =============================================================================
# TODO:
# 1) Load your image (grayscale) into `img_gray` (2D array).
# 2) Build your image centroid array `img_xy` with shape (N, 2):
#       img_xy[:,0] = x_centroid
#       img_xy[:,1] = y_centroid
#    Suggested: use only the brightest K stars (K=200–500).
#
# 3) Build your catalog array `cat_xy_unit` with shape (M, 2) in UNIT coordinates:
#       horizon radius ~ 1
#       zenith near (0,0)
#    Suggested: use only bright catalog stars (e.g., G < 10 or top M=1000).
#
# 4) Set image geometry:
#       cx, cy = fisheye center in pixels
#       R_pix  = horizon radius in pixels
#
# Notes:
# - `cat_xy_unit` should already be in a zenith-centered coordinate system where
#   the horizon is at r≈1. (You made this in the Gaia/AltAz projection step.)
# - If your image uses a different coordinate convention, don't worry: we will
#   solve rotation/flip by searching.
#
# Example placeholders:
image_path = Path(r"Testing Images\256_251029_204008_1761770474.jpg")
go = GONetFile.from_file(image_path)
go.remove_overscan()
img_gray = go.green
#img_xy = None          # TODO: np.array shape (N,2)
img_xy = compute_img_centroids(image_path, N=10, top_k=300)
#cat_xy_unit = None     # TODO: np.array shape (M,2)
cat_xy_unit = compute_cat_unit_coords(image_path, radius_deg=70.0, gmax=5, top_m=300)
#cx, cy = None, None    # TODO: floats
cx, cy = 1030, 760  # Example: center of image
R_pix =  740
plt.imshow(img_gray.data)
plt.plot([cx-R_pix, cx+R_pix], [cy, cy],)
plt.plot([cx, cx], [cy-R_pix, cy+R_pix])
plt.show()
#R_pix = None           # TODO: float

# Quick safety checks (keep these)
def _check_inputs():
    if img_gray is None:
        raise ValueError("img_gray is None. Load your image into img_gray.")
    if img_xy is None or not isinstance(img_xy, np.ndarray) or img_xy.ndim != 2 or img_xy.shape[1] != 2:
        raise ValueError("img_xy must be a NumPy array of shape (N,2).")
    if cat_xy_unit is None or not isinstance(cat_xy_unit, np.ndarray) or cat_xy_unit.ndim != 2 or cat_xy_unit.shape[1] != 2:
        raise ValueError("cat_xy_unit must be a NumPy array of shape (M,2).")
    if cx is None or cy is None or R_pix is None:
        raise ValueError("Set cx, cy, and R_pix (image center and radius).")

# TODO: uncomment once you've filled inputs
_check_inputs()


# =============================================================================
# STEP 1 — Helper: rotation matrix
# =============================================================================
def rot2d(theta_rad: float) -> np.ndarray:
    """
    Create a 2D rotation matrix.

    Parameters
    ----------
    theta_rad : float
        Angle in radians.

    Returns
    -------
    R : np.ndarray
        2x2 rotation matrix.
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)


# =============================================================================
# STEP 2 — Transform catalog unit coords -> image pixel coords
# =============================================================================
def transform_catalog_to_pixels(
    cat_xy_unit: np.ndarray,
    cx: float, cy: float, R_pix: float,
    theta_deg: float,
    flip_x: bool = False,
    flip_y: bool = False,
) -> np.ndarray:
    """
    Apply flips + rotation to catalog unit coords, then scale/shift to pixels.

    Think of this as:
      unit coords (rough sky)  -->  pixel coords (camera)

    Parameters
    ----------
    cat_xy_unit : (M,2) np.ndarray
        Catalog points in unit-disk coordinates (horizon ~ r=1).
    cx, cy : float
        Image center in pixels.
    R_pix : float
        Image radius in pixels (horizon).
    theta_deg : float
        Rotation angle (degrees).
    flip_x, flip_y : bool
        Mirror flips in unit coordinate system.

    Returns
    -------
    cat_xy_pix : (M,2) np.ndarray
        Catalog points mapped into pixel coordinates.
    """
    xy = cat_xy_unit.copy()

    # Optional flips (mirror)
    if flip_x:
        xy[:, 0] *= -1
    if flip_y:
        xy[:, 1] *= -1

    # Rotation
    R = rot2d(np.deg2rad(theta_deg))
    xy = xy @ R.T

    # Scale + shift into pixels
    out = np.empty_like(xy)
    out[:, 0] = cx + R_pix * xy[:, 0]
    out[:, 1] = cy + R_pix * xy[:, 1]
    return out


# =============================================================================
# STEP 3 — Define a matching score (nearest neighbor)
# =============================================================================
def nearest_neighbor_distances(img_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """
    For each point in query_xy, find distance to the nearest point in img_xy.

    This is the core of our "matching" idea:
      - If catalog points land near detected image stars -> good.
      - If they land far away -> bad.

    Returns
    -------
    dists : (M,) np.ndarray
        Nearest-neighbor distance (pixels) for each query point.
    """
    if HAVE_SCIPY:
        tree = cKDTree(img_xy)
        dists, _ = tree.query(query_xy, k=1)
        return dists

    # Fallback (slow, but works)
    dists = np.empty(query_xy.shape[0], dtype=float)
    for i, p in enumerate(query_xy):
        d = np.sqrt(((img_xy - p) ** 2).sum(axis=1))
        dists[i] = d.min()
    return dists


def score_transform(img_xy: np.ndarray, cat_xy_pix: np.ndarray, tol_pix: float) -> int:
    """
    Simple score: how many catalog points have a nearby image centroid within tol_pix?

    Higher score = better.

    Parameters
    ----------
    tol_pix : float
        Matching tolerance in pixels.

    Returns
    -------
    score : int
        Number of catalog points that are within tol_pix of some image centroid.
    """
    dists = nearest_neighbor_distances(img_xy, cat_xy_pix)
    return int(np.sum(dists <= tol_pix))


# =============================================================================
# STEP 4 — Guided: choose good subsets (VERY important)
# =============================================================================
# Before searching, choose manageable sets:
#
# TODO (strong suggestion):
# - Keep only top K image stars (brightest by flux).
# - Keep only top M catalog stars (brightest by magnitude).
#
# Why?
# - Faint detections include noise and blends.
# - Faint catalog stars are too many and make matching harder.
#
# Suggested starting values:
#   K = 200 to 500
#   M = 500 to 2000
#
# If you already filtered, just proceed.


# =============================================================================
# STEP 5 — Search rotation + flips
# =============================================================================
def search_best_orientation(
    img_xy: np.ndarray,
    cat_xy_unit: np.ndarray,
    cx: float, cy: float, R_pix: float,
    tol_pix: float = 20.0,
    step_deg: float = 2.0
) -> dict:
    """
    Search over rotation angle and flips to maximize the match score.

    Parameters
    ----------
    tol_pix : float
        How close (in pixels) a catalog point must be to count as a match.
    step_deg : float
        Rotation step size. Start coarse (2–5 deg), then refine later.

    Returns
    -------
    best : dict
        Contains best theta, flips, score, and the transformed catalog points.
    """
    thetas = np.arange(0.0, 360.0, step_deg)
    best = {"score": -1}

    for flip_x in (False, True):
        for flip_y in (False, True):
            for theta in thetas:
                cat_xy_pix = transform_catalog_to_pixels(
                    cat_xy_unit, cx, cy, R_pix,
                    theta_deg=float(theta),
                    flip_x=flip_x,
                    flip_y=flip_y
                )
                s = score_transform(img_xy, cat_xy_pix, tol_pix=tol_pix)
                if s > best["score"]:
                    best = {
                        "score": s,
                        "theta_deg": float(theta),
                        "flip_x": flip_x,
                        "flip_y": flip_y,
                        "cat_xy_pix": cat_xy_pix
                    }

    return best


# =============================================================================
# STEP 6 — Visual check (overlay plot)
# =============================================================================
def plot_overlay(
    img_gray: np.ndarray,
    img_xy: np.ndarray,
    cat_xy_pix: np.ndarray,
    tol_pix: float = 20.0,
    title: str = ""
) -> None:
    """
    Overlay detected image centroids and transformed catalog points on the image.

    We color catalog points by whether they are near an image centroid.
    """
    dists = nearest_neighbor_distances(img_xy, cat_xy_pix)
    good = dists <= tol_pix

    plt.figure(figsize=(9, 9))
    plt.imshow(img_gray, origin="lower")

    # Image detections
    plt.scatter(
        img_xy[:, 0], img_xy[:, 1],
        s=15, edgecolors="lime", facecolors="none",
        label="Image centroids"
    )

    # Catalog transformed points
    plt.scatter(
        cat_xy_pix[~good, 0], cat_xy_pix[~good, 1],
        s=12, edgecolors="red", facecolors="none",
        label="Catalog (unmatched)"
    )
    plt.scatter(
        cat_xy_pix[good, 0], cat_xy_pix[good, 1],
        s=22, edgecolors="cyan", facecolors="none",
        label="Catalog (matched)"
    )

    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.show()


# =============================================================================
# STEP 7 — Main guided run
# =============================================================================
def main():
    # TODO: Once you filled inputs, enable checks:
    _check_inputs()

    # --- Choose tolerances and search parameters ---
    # TODO:
    # Start with a generous tolerance (e.g., 20–40 px), because our model is imperfect.
    # Start with coarse steps (2–5 degrees), then refine later.
    tol_pix = 25.0
    step_deg = 3.0

    # --- Quick sanity plot of detections alone ---
    # TODO (optional): plot your image with just img_xy to confirm they look reasonable.
    plt.figure(figsize=(8, 8))
    plt.imshow(img_gray, origin="lower")
    plt.scatter(img_xy[:, 0], img_xy[:, 1], s=15, edgecolors="lime", facecolors="none")
    plt.title("Sanity check: image centroids")
    plt.axis("off")
    plt.show()

    # --- Search best orientation ---
    print("Searching rotation + flips...")
    best = search_best_orientation(
        img_xy=img_xy,
        cat_xy_unit=cat_xy_unit,
        cx=float(cx), cy=float(cy), R_pix=float(R_pix),
        tol_pix=tol_pix,
        step_deg=step_deg
    )

    print("\n=== Best orientation (coarse search) ===")
    print("Score (matches within tol):", best["score"])
    print("theta_deg:", best["theta_deg"])
    print("flip_x:", best["flip_x"])
    print("flip_y:", best["flip_y"])
    print("======================================\n")

    # --- Plot overlay result ---
    title = f"Best (coarse): score={best['score']}  theta={best['theta_deg']}  flip_x={best['flip_x']}  flip_y={best['flip_y']}"
    plot_overlay(img_gray, img_xy, best["cat_xy_pix"], tol_pix=tol_pix, title=title)

    # --- Optional refinement around the best theta ---
    # TODO:
    # Once you have a good coarse theta, refine around it:
    #   - search theta in [best_theta - 5°, best_theta + 5°] with step 0.2°
    #   - keep the same flips
    #
    # This makes the overlay tighter.
    #
    # best_theta = best["theta_deg"]
    # flip_x = best["flip_x"]
    # flip_y = best["flip_y"]
    # fine_thetas = np.arange(best_theta - 5, best_theta + 5.0001, 0.2)
    # best_fine = {"score": -1}
    #
    # for theta in fine_thetas:
    #     cat_xy_pix = transform_catalog_to_pixels(cat_xy_unit, cx, cy, R_pix,
    #                                              theta_deg=float(theta),
    #                                              flip_x=flip_x, flip_y=flip_y)
    #     s = score_transform(img_xy, cat_xy_pix, tol_pix=tol_pix)
    #     if s > best_fine["score"]:
    #         best_fine = {"score": s, "theta_deg": float(theta), "cat_xy_pix": cat_xy_pix}
    #
    # print("Best fine theta:", best_fine["theta_deg"], "score:", best_fine["score"])
    # plot_overlay(img_gray, img_xy, best_fine["cat_xy_pix"], tol_pix=tol_pix,
    #              title=f"Best (fine): score={best_fine['score']} theta={best_fine['theta_deg']:.2f}")

    # -------------------------------------------------------------------------
    # EXTRA (optional): WCS fit once you have matches
    # -------------------------------------------------------------------------
    # If your overlay looks decent, you can attempt an Astropy WCS fit.
    #
    # What you need:
    # - matched pairs: (x_img, y_img) <-> (RA, Dec)
    #
    # You can create matches like this:
    # 1) You already have cat_xy_pix from the best transform
    # 2) Find nearest image centroid for each catalog point
    # 3) Keep those with distance < tol_pix
    #
    # Then fit:
    #   from astropy.wcs.utils import fit_wcs_from_points
    #   wcs = fit_wcs_from_points((x_img, y_img), matched_skycoords, projection="TAN")
    #
    # TODO (optional):
    # - Store the sky coords for your catalog stars aligned with cat_xy_unit
    #   (e.g., cat_sky = SkyCoord(ra=..., dec=...))
    # - Create matched_sky and matched_img arrays based on nearest neighbors
    # - Fit WCS and compute residuals
    #
    # We are leaving this optional so everyone can get to the end today.


if __name__ == "__main__":
    main()