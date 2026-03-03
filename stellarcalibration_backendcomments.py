"""
star_calibration_backend.py
============================
This module contains all the science/math logic for the Star Calibration tool.
It takes a GONet all-sky camera image, detects stars in the image, downloads
bright star positions from the Gaia star catalogue, figures out where the
camera is pointing, and then shifts the image so that the zenith (the point
directly overhead) ends up exactly in the centre of the image.

High-level pipeline:
  1. Load the image and detect bright spots (stars) using pixel thresholding.
  2. Download catalog star positions (Alt/Az angles) from the Gaia database.
  3. Run a grid search to find the camera orientation that best aligns catalog
     stars with detected image stars.
  4. Fit a World Coordinate System (WCS) to the matched star pairs so we know
     exactly which pixel corresponds to which sky position.
  5. Find where the zenith falls in the image and shift the whole image so
     that point moves to the image centre.
  6. Return the shifted image (and diagnostic info) to the GUI.
"""

import numpy as np

# cKDTree is a fast spatial data structure for nearest-neighbour searches.
# We use it to quickly find which detected image star is closest to each
# predicted catalog-star position without looping through every pair manually.
from scipy.spatial import cKDTree

# scipy.ndimage provides fast C-level image-processing routines:
#   nd_label         — finds connected groups of bright pixels (star blobs)
#   ndimage_sum      — sums pixel values inside each labelled blob
#   ndimage_com      — calculates the centre-of-mass of each labelled blob
#   nd_shift         — slides an image by a sub-pixel amount
from scipy.ndimage import (
    label          as nd_label,
    sum            as ndimage_sum,
    center_of_mass as ndimage_com,
    shift          as nd_shift,
)

# GONetFile understands the proprietary GONet camera file format and exposes
# useful properties like .green (the green pixel channel) and .meta (metadata
# containing GPS position and capture time).
from GONet_Wizard.GONet_utils import GONetFile

# Pillow (PIL) is used to open/save images and convert between formats.
from PIL import Image

# astropy provides astronomy-specific handling of angle units, times,
# sky coordinates, and World Coordinate System (WCS) fitting.
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.wcs.utils import fit_wcs_from_points

# astroquery lets us make live queries to online astronomical databases.
# Here we use the Gaia DR3 catalogue of ~1.8 billion stars.
from astroquery.gaia import Gaia


# =============================================================================
# STEP 1 — STAR DETECTION IN THE IMAGE
# =============================================================================

# Replaced pure-Python BFS with scipy.ndimage.label (C implementation; ~10-100x faster).
def compact_search(mask):
    """
    Find all connected groups of bright pixels (star blobs) in the image.

    A "mask" is a 2-D grid of True/False values: True where a pixel is
    brighter than our detection threshold, False everywhere else.

    scipy's nd_label scans the mask and assigns every connected group of True
    pixels the same integer label (1, 2, 3, …).  This is equivalent to
    finding all separate star blobs in the image.

    Parameters
    ----------
    mask : 2-D boolean numpy array
        True wherever a pixel is above the brightness threshold.

    Returns
    -------
    labels : 2-D int array
        Same shape as mask; each pixel holds the ID of its blob (0 = background).
    num_clusters : int
        Total number of distinct blobs found.
    """
    labels, num_clusters = nd_label(mask)
    return labels, num_clusters


# Replaced per-cluster Python loops with vectorized scipy.ndimage bulk functions.
def measure_sources(sub, labels, num_clusters):
    """
    Measure the position and brightness of every detected star blob.

    For each numbered blob we compute:
      - total_flux  : sum of all pixel values inside the blob (brightness proxy)
      - centroid    : flux-weighted centre position (x, y) — more accurate than
                      just the peak pixel because it uses all the blob's pixels.

    We discard blobs with zero or negative total flux (hot-pixel artefacts,
    over-subtracted background, etc.).

    Parameters
    ----------
    sub : 2-D float array
        The image pixel data (we use the green Bayer channel).
    labels : 2-D int array
        Label array returned by compact_search.
    num_clusters : int
        Number of blobs to measure.

    Returns
    -------
    sources : list of dicts
        One dict per valid source containing label, centroid x/y, and flux.
    x_centroids : list of float
        x (column) centroid positions for every valid source.
    y_centroids : list of float
        y (row) centroid positions for every valid source.
    """
    # Nothing to do if no blobs were found.
    if num_clusters == 0:
        return [], [], []

    label_ids = np.arange(1, num_clusters + 1)  # [1, 2, 3, …, num_clusters]

    # ndimage_sum  → 1-D array: total brightness of each blob.
    # ndimage_com  → list of (y, x) tuples: flux-weighted centroid of each blob.
    # Both run in vectorised C code — no slow Python loop over individual blobs.
    total_fluxes = ndimage_sum(sub, labels, index=label_ids)
    centers      = ndimage_com(sub, labels, index=label_ids)

    sources     = []
    x_centroids = []
    y_centroids = []

    for i, label_id in enumerate(label_ids):
        tf = float(total_fluxes[i])

        # Skip blobs that have no positive signal (spurious detections).
        if tf <= 0:
            continue

        # ndimage_com returns (row, col) i.e. (y, x) — unpack accordingly.
        y_c, x_c = centers[i]

        sources.append({
            "label":      int(label_id),
            "x_centroid": float(x_c),
            "y_centroid": float(y_c),
            "flux":       tf,
        })
        x_centroids.append(float(x_c))
        y_centroids.append(float(y_c))

    return sources, x_centroids, y_centroids



# =============================================================================
# STEP 2 — DOWNLOAD STAR CATALOG POSITIONS
# =============================================================================

def query_catalog_altaz_from_meta(meta, radius_deg=60.0, gmax=2.5, top_m=None):
    """
    Download bright star positions from the Gaia DR3 catalogue and convert
    them to Altitude/Azimuth (Alt/Az) sky coordinates as seen from the camera.

    "Altitude" is the angle above the horizon (0° = horizon, 90° = zenith).
    "Azimuth"  is the compass bearing measured clockwise from North.

    We only download stars within `radius_deg` of the zenith and brighter
    than Gaia G-band magnitude `gmax` (lower number = brighter star).

    Parameters
    ----------
    meta : dict
        Camera metadata from the GONet file.  Must contain:
          meta["GPS"]["latitude"]   — decimal degrees north
          meta["GPS"]["longitude"]  — decimal degrees east
          meta["GPS"]["altitude"]   — metres above sea level
          meta["DateTime"]          — "YYYY:MM:DD HH:MM:SS" UTC string
    radius_deg : float
        Angular radius around zenith to search (default 60°).
    gmax : float
        Maximum (faintest) Gaia G magnitude to include (default 2.5 = very bright).
    top_m : int or None
        If given, keep only the `top_m` brightest stars after filtering.

    Returns
    -------
    alt  : numpy array  — altitude angles in degrees (above horizon).
    az   : numpy array  — azimuth angles in degrees.
    gmag : numpy array  — Gaia G-band magnitudes.
    """
    # --- Parse the observer's location from the image metadata ---------------
    lat_deg = float(meta["GPS"]["latitude"])
    lon_deg = float(meta["GPS"]["longitude"])
    alt_m   = float(meta["GPS"]["altitude"])

    # Convert the stored "YYYY:MM:DD HH:MM:SS" string to ISO 8601 format
    # "YYYY-MM-DDTHH:MM:SS" which astropy's Time parser understands.
    # The first two colons (in the date part) are replaced with hyphens;
    # then the space between date and time is replaced with 'T'.
    ut_iso = meta["DateTime"].replace(":", "-", 2).replace(" ", "T")

    # Build astropy objects representing the physical location on Earth
    # and the exact moment of the observation.
    location = EarthLocation(lat=lat_deg * u.deg,
                             lon=lon_deg * u.deg,
                             height=alt_m * u.m)
    obstime  = Time(ut_iso, scale="utc")

    # --- Find the zenith in sky coordinates (RA/Dec) -------------------------
    # The zenith is Alt=90°, Az=0° in the local Alt/Az frame.
    # We convert it to the fixed ICRS frame (RA/Dec) so we can query Gaia’s
    # catalogue, which stores star positions in RA/Dec, not local Alt/Az.
    zenith_altaz = SkyCoord(
        alt=90 * u.deg,
        az=0  * u.deg,
        frame=AltAz(obstime=obstime, location=location),
    )
    zenith_icrs = zenith_altaz.icrs   # now expressed as (RA, Dec)

    # --- Query Gaia for bright stars near the zenith -------------------------
    # We use an ADQL (Astronomical Data Query Language) query — essentially
    # SQL for astronomy databases.  The CIRCLE function selects all stars
    # inside a cone of `radius_deg` degrees centred on the zenith RA/Dec.
    Gaia.ROW_LIMIT = 200000           # safety cap: never download more than 200 k rows
    ra0  = zenith_icrs.ra.deg
    dec0 = zenith_icrs.dec.deg

    query = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra0}, {dec0}, {radius_deg})
    )
    AND phot_g_mean_mag < {gmax}
    """

    # launch_job_async sends the query to the Gaia archive server and waits
    # for the result.  This network call typically takes a few seconds.
    job = Gaia.launch_job_async(query)
    tbl = job.get_results()

    # --- Convert Gaia RA/Dec positions to local Alt/Az -----------------------
    # We rotate from the fixed sky frame (RA/Dec) into the observer's local
    # frame (Alt/Az) so we can later project the stars onto the image.
    stars_icrs  = SkyCoord(ra  = np.array(tbl["ra"])  * u.deg,
                           dec = np.array(tbl["dec"]) * u.deg,
                           frame="icrs")
    stars_altaz = stars_icrs.transform_to(
        AltAz(obstime=obstime, location=location)
    )

    alt  = stars_altaz.alt.deg
    az   = stars_altaz.az.deg
    gmag = np.array(tbl["phot_g_mean_mag"])

    # Keep only stars that are actually above the horizon (alt > 0°).
    # Stars below the horizon are blocked by the Earth and cannot be seen.
    above = alt > 0
    alt, az, gmag = alt[above], az[above], gmag[above]

    # Optionally restrict to the N brightest stars.
    if top_m is not None and len(gmag) > top_m:
        idx  = np.argsort(gmag)[:top_m]   # argsort returns indices from lowest to highest
        alt, az, gmag = alt[idx], az[idx], gmag[idx]

    return alt, az, gmag


# =============================================================================
# STEP 3 — CAMERA ORIENTATION MATH
# =============================================================================
# The camera has a fisheye (all-sky) lens.  To figure out which pixel any
# given star appears on, we need to know three things:
#   alpha — which compass direction the "top" of the image faces (rotation
#            around the vertical axis, like turning a compass).
#   beta  — how far the camera is tilted away from perfectly vertical.
#   gamma — the "roll" angle: the compass direction the camera tilts toward.
#
# We represent these rotations with 3×3 rotation matrices.  Multiplying a
# direction vector by a rotation matrix applies that rotation to the vector.

def unitvec_from_altaz(alt_rad, az_rad):
    """
    Convert (altitude, azimuth) angles to a 3-D unit vector in the local frame.

    Convention:
      x-axis = East
      y-axis = North
      z-axis = Up (toward zenith)

    A unit vector is just a direction arrow of length 1.  We use vectors
    instead of angles because matrix multiplication on vectors is what lets
    us apply rotations efficiently.

    Parameters
    ----------
    alt_rad : array-like  — altitude angle(s) in radians.
    az_rad  : array-like  — azimuth angle(s) in radians.

    Returns
    -------
    numpy array, shape (N, 3) — one unit vector per input angle pair.
    """
    ca = np.cos(alt_rad)              # horizontal component of the direction
    x  = ca * np.sin(az_rad)          # East component
    y  = ca * np.cos(az_rad)          # North component
    z  = np.sin(alt_rad)              # Up component
    return np.stack([x, y, z], axis=-1)


def rot_z(angle_rad):
    """
    Build a 3×3 matrix that rotates vectors around the Z-axis (vertical).
    This is equivalent to rotating a compass — spinning things in the
    horizontal plane.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], float)


def rot_x(angle_rad):
    """
    Build a 3×3 matrix that rotates vectors around the X-axis (East).
    This tilts things forward/backward, like nodding your head.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]], float)


def orientation_matrix(alpha, beta, gamma):
    """
    Combine the three camera orientation angles into one rotation matrix R.

    R transforms a sky direction vector (in the local North/East/Up frame)
    into a camera-frame vector (pointing into the camera's lens).

    Parameters
    ----------
    alpha : float  — azimuthal heading of the camera image "up" direction (radians)
    beta  : float  — tilt of the camera away from vertical (radians)
    gamma : float  — compass direction toward which the camera tilts (radians)

    Returns
    -------
    R : 3×3 numpy float array
    """
    R_alpha = rot_z(alpha)
    # The tilt is applied as: rotate to face the tilt direction (rot_z(gamma)),
    # apply the tilt (rot_x(beta)), then rotate back (rot_z(-gamma)).
    R_tilt  = rot_z(gamma) @ rot_x(beta) @ rot_z(-gamma)
    return R_tilt @ R_alpha



# =============================================================================
# STEP 4 — FISHEYE LENS PROJECTION
# =============================================================================

def r_from_theta(theta_rad, R_pix):
    """
    Convert an angular distance from the optical axis (theta) to a pixel radius.

    This implements a simple equidistant fisheye projection:
      r = (theta / 90°) × R_pix
    where R_pix is the pixel radius corresponding to the horizon (theta = 90°).
    A star directly overhead (theta = 0) maps to the image centre (r = 0).
    A star at the horizon (theta = 90°) maps to r = R_pix.

    Parameters
    ----------
    theta_rad : float or array  — angle from image centre (optical axis) in radians.
    R_pix     : float           — pixel radius of the horizon circle in the image.

    Returns
    -------
    float or array  — pixel distance from the image centre.
    """
    return (theta_rad / (np.pi / 2)) * R_pix


def filter_image_sources_by_radius(img_xy, cx, cy, R_pix, radius_deg):
    """
    Remove detected image sources that fall outside the sky region covered by
    the star catalogue query.

    If we queried Gaia for stars within 60° of zenith, we should also discard
    any image detections more than 60° from zenith (i.e. near the horizon),
    because those can never match a catalog star.

    Parameters
    ----------
    img_xy     : (N, 2) array  — pixel (x, y) positions of detected sources.
    cx, cy     : float         — optical centre of the fisheye lens in pixels.
    R_pix      : float         — pixel radius of the horizon (90° from zenith).
    radius_deg : float         — angular cap used for the Gaia query.

    Returns
    -------
    (M, 2) array  — subset of img_xy inside the angular cap (M ≤ N).
    """
    if len(img_xy) == 0:
        return img_xy

    # Convert the angular cap from degrees to pixels using the fisheye model.
    max_r_pix = (float(radius_deg) / 90.0) * float(R_pix)

    # Calculate each source's pixel distance from the lens centre.
    dx = img_xy[:, 0] - float(cx)
    dy = img_xy[:, 1] - float(cy)
    rr = np.sqrt(dx * dx + dy * dy)

    # Boolean mask: True for sources inside the allowed radius.
    keep = rr <= max_r_pix
    return img_xy[keep]


# =============================================================================
# STEP 5 — FORWARD MODEL: CATALOG POSITIONS → PREDICTED PIXELS
# =============================================================================

def predict_pixels_from_catalog(alt_deg, az_deg, cx, cy, R_pix,
                                 alpha, beta, gamma):
    """
    Given a camera orientation guess (alpha, beta, gamma) and the physical
    lens parameters (cx, cy, R_pix), project every catalog star's Alt/Az sky
    position onto the image plane and return predicted pixel coordinates.

    This is the "forward model": given a guess at the camera orientation,
    where would each star appear on the sensor?

    Parameters
    ----------
    alt_deg, az_deg         : arrays  — catalog star altitudes and azimuths (degrees).
    cx, cy                  : float   — optical centre of the fisheye lens (pixels).
    R_pix                   : float   — pixel radius of the horizon.
    alpha, beta, gamma      : float   — camera orientation angles (radians).

    Returns
    -------
    (N, 2) float array  — predicted (x, y) pixel positions for each catalog star.
    """
    alt = np.deg2rad(alt_deg)
    az  = np.deg2rad(az_deg)

    # 1. Convert catalog Alt/Az to unit vectors in the sky (North/East/Up) frame.
    v_sky = unitvec_from_altaz(alt, az)

    # 2. Rotate those sky vectors into the camera's own coordinate frame.
    #    v_cam[:,2] is the "depth" component — how far along the optical axis
    #    each star direction lies.
    R     = orientation_matrix(alpha, beta, gamma)
    v_cam = v_sky @ R.T                   # shape (N, 3)

    # 3. theta = angle between the star direction and the camera's optical axis.
    #    Stars directly ahead have theta=0; horizon stars have theta=90°.
    z     = np.clip(v_cam[:, 2], -1, 1)   # clamp to [-1, 1] to avoid arccos errors
    theta = np.arccos(z)

    # 4. phi = angle around the optical axis (like a compass bearing in the image).
    phi = np.arctan2(v_cam[:, 1], v_cam[:, 0])

    # 5. Apply the fisheye lens model to get the pixel distance from image centre.
    r = r_from_theta(theta, R_pix)

    # 6. Convert polar (r, phi) → Cartesian (x, y) pixel coordinates.
    x = cx + r * np.cos(phi)
    y = cy + r * np.sin(phi)
    return np.stack([x, y], axis=-1)


# =============================================================================
# STEP 6 — MATCHING SCORE
# =============================================================================

def match_score(img_tree, pred_xy, tol_pix=20.0):
    """
    Count how many predicted catalog-star positions land close to a detected
    image star.

    "Close" means within `tol_pix` pixels.  A higher score means the current
    camera-orientation guess is a better fit to the data.

    Using a pre-built cKDTree for img_xy (passed in from solve_orientation)
    avoids rebuilding the spatial index on every iteration of the grid search,
    which would be thousands of redundant builds.

    Parameters
    ----------
    img_tree : cKDTree
        Spatial index of detected image-star positions, built ONCE outside
        this function and reused for every call.
    pred_xy  : (N, 2) array  — predicted pixel positions for each catalog star.
    tol_pix  : float         — matching tolerance in pixels.

    Returns
    -------
    score : int          — number of catalog stars matched to an image star.
    dists : (N,) array   — distance to the nearest image star for every catalog star.
    idx   : (N,) int array — index of the nearest image star for every catalog star.
    """
    # query(pred_xy, k=1) finds the single nearest image star for each
    # predicted position and returns its distance and list index.
    dists, idx = img_tree.query(pred_xy, k=1)
    score = np.sum(dists <= tol_pix)
    return int(score), dists, idx



# =============================================================================
# STEP 7 — ORIENTATION SOLVING (COARSE GRID SEARCH + FINE REFINEMENT)
# =============================================================================

def solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix):
    """
    Find the camera orientation angles (alpha, beta, gamma) that best explain
    where the catalog stars appear in the image.

    Strategy — two-pass grid search:
      Pass 1 (coarse): Try every combination on a wide, coarse grid:
          alpha every 5° (72 values), beta every 2° up to 10°, gamma every 20°.
          This covers all possible orientations but with low precision.
      Pass 2 (fine): Zoom in around the best coarse result on a tighter grid:
          alpha ±5° in 0.5° steps, beta ±2° in 0.2° steps, gamma ±10° in 1° steps.

    The cKDTree is built ONCE here before the loops and passed into every
    match_score call, avoiding thousands of redundant tree constructions.

    Parameters
    ----------
    img_xy       : (N, 2) array  — detected star positions in the image (pixels).
    cat_alt_deg  : array         — catalog star altitudes (degrees).
    cat_az_deg   : array         — catalog star azimuths (degrees).
    cx, cy       : float         — optical centre of the lens (pixels).
    R_pix        : float         — pixel radius of the horizon.

    Returns
    -------
    best : dict
        Keys: score, alpha, beta, gamma, dists, idx, pred_xy,
              matched_count, rms_pix.
    """
    # Build the KDTree once — this is the spatial index of all detected image
    # stars.  Every match_score call then just queries this same tree.
    img_tree = cKDTree(img_xy)

    # --- Coarse grid pass ---------------------------------------------------
    alpha_grid = np.deg2rad(np.arange(0, 360, 5))    # 72 values covering full rotation
    beta_grid  = np.deg2rad(np.arange(0, 11, 2))     # 6 values: 0° to 10° tilt
    gamma_grid = np.deg2rad(np.arange(0, 360, 20))   # 18 values covering tilt direction

    best = {"score": -1}   # start with an impossibly bad score

    for beta in beta_grid:
        # When the camera is perfectly vertical (beta=0), any value of gamma
        # produces the same rotation, so we only need to try gamma=0.
        gamma_list = [0.0] if abs(beta) < 1e-12 else gamma_grid
        for gamma in gamma_list:
            for alpha in alpha_grid:
                pred_xy = predict_pixels_from_catalog(
                    cat_alt_deg, cat_az_deg, cx, cy, R_pix, alpha, beta, gamma
                )
                s, dists, idx = match_score(img_tree, pred_xy, tol_pix=25.0)
                if s > best["score"]:
                    best = {
                        "score": s, "alpha": alpha, "beta": beta,
                        "gamma": gamma, "dists": dists, "idx": idx,
                        "pred_xy": pred_xy,
                    }

    # --- Fine grid pass (zoom in around the best coarse solution) -----------
    alpha_refine = best["alpha"] + np.deg2rad(np.arange(-5.0,  5.0  + 1e-12, 0.5))
    beta_refine  = best["beta"]  + np.deg2rad(np.arange(-2.0,  2.0  + 1e-12, 0.2))
    gamma_refine = best["gamma"] + np.deg2rad(np.arange(-10.0, 10.0 + 1e-12, 1.0))

    # Wrap angles into [0, 2π) so we never accidentally go negative.
    alpha_refine = np.mod(alpha_refine, 2 * np.pi)
    gamma_refine = np.mod(gamma_refine, 2 * np.pi)
    # Clamp beta to a physically meaningful range [0°, 15°].
    beta_refine  = np.clip(beta_refine, 0.0, np.deg2rad(15.0))

    for beta in np.unique(beta_refine):
        gamma_list = [best["gamma"]] if abs(beta) < 1e-12 else np.unique(gamma_refine)
        for gamma in gamma_list:
            for alpha in np.unique(alpha_refine):
                pred_xy = predict_pixels_from_catalog(
                    cat_alt_deg, cat_az_deg, cx, cy, R_pix, alpha, beta, gamma
                )
                s, dists, idx = match_score(img_tree, pred_xy, tol_pix=25.0)
                if s > best["score"]:
                    best = {
                        "score": s, "alpha": alpha, "beta": beta,
                        "gamma": gamma, "dists": dists, "idx": idx,
                        "pred_xy": pred_xy,
                    }

    # --- Summary statistics on the best solution ---------------------------
    matched_mask  = best["dists"] <= 25.0
    matched_count = int(np.sum(matched_mask))

    # RMS residual: the square root of the mean squared distance between each
    # matched catalog-star prediction and its nearest image star.
    # Smaller RMS = tighter, more accurate solution.
    if matched_count > 0:
        best["rms_pix"] = float(
            np.sqrt(np.mean(best["dists"][matched_mask] ** 2))
        )
    else:
        best["rms_pix"] = np.nan    # no matches — RMS is undefined

    best["matched_count"] = matched_count
    return best



# =============================================================================
# STEP 8 — WCS FIT AND ZENITH CENTRING
# =============================================================================

def fit_wcs_and_center_zenith(
    sub,
    img_xy,
    cat_alt_deg,
    cat_az_deg,
    best,
    meta,
    tol_pix=25.0,
    min_wcs_matches=3,
):
    """
    Use the matched star pairs to fit a World Coordinate System (WCS), then
    shift the image so that the zenith lands exactly at the image centre.

    A WCS is a mathematical description of the mapping between pixel coordinates
    and sky coordinates (RA/Dec).  Once we have a WCS, we can ask:
    "Which pixel does the zenith (Alt=90°) appear at?" and then slide the
    image to put that pixel at the centre.

    Steps
    -----
    1. Keep only catalog↔image pairs that are within tol_pix of each other.
    2. Deduplicate: if two catalog stars map to the same image star, keep only
       the closest match (prevents using one detection twice).
    3. Convert matched catalog Alt/Az positions to ICRS (RA/Dec) — the
       coordinate system the WCS fitter requires.
    4. Fit a TAN-projection WCS to the (pixel, RA/Dec) point pairs.
    5. Ask the WCS where Alt=90° appears → (zenith_x, zenith_y) in pixels.
    6. Compute the shift needed to move the zenith to the image centre.
    7. Apply the shift and return diagnostics.

    Parameters
    ----------
    sub          : 2-D array        — raw image channel used for detection.
    img_xy       : (N, 2) array     — detected image-star pixel positions.
    cat_alt_deg  : array            — catalog star altitudes in degrees.
    cat_az_deg   : array            — catalog star azimuths in degrees.
    best         : dict             — result dict from solve_orientation.
    meta         : dict             — GONet file metadata (GPS + time).
    tol_pix      : float            — maximum pixel distance to count a match.
    min_wcs_matches : int           — minimum star pairs needed for a WCS fit.

    Returns
    -------
    dict with keys:
        wcs             — fitted astropy WCS object (or None on failure)
        zenith_x/y      — predicted zenith pixel position (or NaN)
        target_cx/cy    — target image centre pixel
        shift_x/y       — translation applied to centre zenith (pixels)
        centered_sub    — the shifted image array
        n_wcs_matches   — number of star pairs used for the WCS fit
        wcs_fit_success — True if WCS fitting succeeded
        wcs_fit_error   — error message string if fitting failed
    """
    # Convenience: build a consistent "failure" result dict so that every
    # early-exit path returns the same dictionary structure.
    def _failure(msg, n=0):
        tcx = (sub.shape[1] - 1) / 2.0
        tcy = (sub.shape[0] - 1) / 2.0
        return {
            "wcs": None, "zenith_x": np.nan, "zenith_y": np.nan,
            "target_cx": float(tcx), "target_cy": float(tcy),
            "shift_x": 0.0, "shift_y": 0.0,
            "centered_sub": sub.astype(float).copy(),
            "n_wcs_matches": n, "wcs_fit_success": False,
            "wcs_fit_error": msg,
        }

    # --- 1. Filter to matches within tolerance ----------------------------
    # best["dists"] holds the pixel distance from each catalog star's predicted
    # position to its nearest detected image star.
    matched_catalog_mask    = best["dists"] <= tol_pix
    matched_catalog_indices = np.where(matched_catalog_mask)[0]    # catalog row IDs
    matched_image_indices   = best["idx"][matched_catalog_mask]    # image star IDs
    matched_distances       = best["dists"][matched_catalog_mask]

    # --- 2. Deduplicate — one image star can only be used once --------------
    # Sort matches from closest to farthest so we keep the best match first.
    sorted_order   = np.argsort(matched_distances)
    used_image_set = set()
    kept_positions = []
    for pos in sorted_order:
        img_idx = int(matched_image_indices[pos])
        if img_idx in used_image_set:
            continue   # already used this image star; skip
        used_image_set.add(img_idx)
        kept_positions.append(pos)

    min_wcs_matches = max(3, int(min_wcs_matches))
    if len(kept_positions) < min_wcs_matches:
        return _failure(
            f"Not enough unique matches for WCS fit "
            f"({len(kept_positions)} < {min_wcs_matches}).",
            n=len(kept_positions),
        )

    kept_positions          = np.array(kept_positions, dtype=int)
    matched_catalog_indices = matched_catalog_indices[kept_positions]
    matched_image_indices   = matched_image_indices[kept_positions]

    # Pixel coordinates of the matched image stars.
    matched_pixel_x = img_xy[matched_image_indices, 0]
    matched_pixel_y = img_xy[matched_image_indices, 1]

    # --- 3. Rebuild observation context from metadata ---------------------
    lat_deg = float(meta["GPS"]["latitude"])
    lon_deg = float(meta["GPS"]["longitude"])
    alt_m   = float(meta["GPS"]["altitude"])
    ut_iso  = meta["DateTime"].replace(":", "-", 2).replace(" ", "T")

    location = EarthLocation(lat=lat_deg * u.deg,
                             lon=lon_deg * u.deg,
                             height=alt_m * u.m)
    obstime  = Time(ut_iso, scale="utc")

    # Convert the matched catalog stars' Alt/Az → ICRS (RA/Dec) because
    # astropy’s WCS fitter works in the fixed ICRS sky frame.
    matched_altaz = SkyCoord(
        alt = np.array(cat_alt_deg)[matched_catalog_indices] * u.deg,
        az  = np.array(cat_az_deg)[matched_catalog_indices]  * u.deg,
        frame=AltAz(obstime=obstime, location=location),
    )
    matched_icrs = matched_altaz.icrs

    # --- 4. Fit the WCS ---------------------------------------------------
    # The zenith in ICRS coordinates (needed as a candidate projection centre
    # and to locate the zenith pixel after fitting).
    zenith_altaz = SkyCoord(alt=90.0 * u.deg, az=0.0 * u.deg,
                            frame=AltAz(obstime=obstime, location=location))
    zenith_icrs  = zenith_altaz.icrs

    # We try several projection centres because the non-linear WCS optimizer
    # occasionally fails for certain choices.  We stop at the first success.
    proj_center_candidates = [
        "center",                         # use the median RA/Dec of the points
        zenith_icrs,                       # zenith direction
        matched_icrs[0],                   # first matched star
        matched_icrs[len(matched_icrs) // 2],  # middle matched star
    ]

    fitted_wcs     = None
    last_wcs_error = None

    for proj_center in proj_center_candidates:
        try:
            fitted_wcs = fit_wcs_from_points(
                (matched_pixel_x, matched_pixel_y),
                matched_icrs,
                projection="TAN",
                proj_point=proj_center,
            )
            break   # success — stop trying other centres
        except Exception as e:
            last_wcs_error = e

    if fitted_wcs is None:
        return _failure(str(last_wcs_error), n=len(kept_positions))

    # --- 5. Find the zenith pixel from the fitted WCS ---------------------
    # to_pixel() is the inverse of what WCS does: it converts a sky coordinate
    # (RA/Dec) back to a pixel (x, y) position using the fitted WCS.
    zenith_pixel_x, zenith_pixel_y = zenith_icrs.to_pixel(fitted_wcs, origin=0)

    # --- 6. Compute the required shift ------------------------------------
    # The "target" centre is simply the central pixel of the image.
    target_cx = (sub.shape[1] - 1) / 2.0
    target_cy = (sub.shape[0] - 1) / 2.0

    # shift_x/y is how many pixels the image must slide so that the zenith
    # pixel moves exactly onto the image centre.
    shift_x = target_cx - float(zenith_pixel_x)
    shift_y = target_cy - float(zenith_pixel_y)

    # --- 7. Apply the shift to the detection sub-image -------------------
    # We use bilinear interpolation (order=1) for smooth sub-pixel shifts.
    # Exposed border pixels (where the image slides out) are filled with the
    # median background brightness so the edge looks natural, not black.
    centered_sub = nd_shift(
        sub.astype(float),
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=float(np.median(sub)),
    )

    return {
        "wcs":             fitted_wcs,
        "zenith_x":        float(zenith_pixel_x),
        "zenith_y":        float(zenith_pixel_y),
        "target_cx":       float(target_cx),
        "target_cy":       float(target_cy),
        "shift_x":         float(shift_x),
        "shift_y":         float(shift_y),
        "centered_sub":    centered_sub,
        "n_wcs_matches":   int(len(kept_positions)),
        "wcs_fit_success": True,
        "wcs_fit_error":   "",
    }



# =============================================================================
# STEP 9 — APPLY SHIFT TO THE ORIGINAL IMAGE FILE
# =============================================================================

def build_shifted_image_same_format(image_path, shift_x, shift_y):
    """
    Re-open the original image file, apply the computed (shift_x, shift_y)
    translation to every channel, and return the result as a PIL Image.

    We shift the *full-quality* original file (not the green-channel detection
    sub-image) so the output retains all original colour channels and bit-depth.
    This function does NOT save the file — the GUI decides where to save it.

    Supports any bit-depth that Pillow can open:
      - 8-bit JPG/PNG (uint8)
      - 16-bit TIFF  (uint16)
      - 32-bit float TIFF (float32)
      - Multi-channel (RGB, RGBA) images

    Parameters
    ----------
    image_path : str or Path
        Path to the original image file.
    shift_x : float
        Horizontal translation in pixels (positive = shift image to the right).
    shift_y : float
        Vertical translation in pixels (positive = shift image downward).

    Returns
    -------
    dict with keys:
        shifted_image    — PIL Image of the translated image (same bit-depth as input).
        shifted_format   — format string for PIL saving (e.g. "TIFF", "JPEG").
        suggested_suffix — file extension to pre-fill in the save dialog.
    """
    pil_image       = Image.open(image_path)
    original_mode   = pil_image.mode
    # pil_image.format is set by Pillow when reading from disk (e.g. "TIFF", "JPEG").
    # It may be None for in-memory images, so we fall back to "PNG" as a safe default.
    original_format = pil_image.format or "PNG"
    original_suffix = (str(image_path).rsplit(".", 1)[-1].lower()
                       if "." in str(image_path) else "png")

    image_array    = np.array(pil_image)
    original_dtype = image_array.dtype

    # Keyword arguments shared by all nd_shift calls.
    shift_kwargs = dict(order=1, mode="constant")

    if image_array.ndim == 2:
        # Greyscale image (or single-channel 16-bit TIFF).
        # Fill the exposed border pixels with the median background value.
        cval    = float(np.median(image_array))
        shifted = nd_shift(image_array.astype(float),
                           shift=(float(shift_y), float(shift_x)),
                           cval=cval, **shift_kwargs)

    elif image_array.ndim == 3:
        # Colour image with shape (Height, Width, Channels).
        # We shift each colour channel (R, G, B, and optionally A) independently
        # using the same shift amount so they all move together.
        shifted_channels = []
        for ch in range(image_array.shape[2]):
            channel = image_array[..., ch]
            cval    = float(np.median(channel))
            shifted_channels.append(
                nd_shift(channel.astype(float),
                         shift=(float(shift_y), float(shift_x)),
                         cval=cval, **shift_kwargs)
            )
        shifted = np.stack(shifted_channels, axis=-1)

    else:
        raise ValueError(
            f"Unsupported image array shape {image_array.shape}.  "
            "Expected 2-D (greyscale) or 3-D (colour) array."
        )

    # Restore the original data type.
    # For integer images (uint8, uint16, …) we clip to the valid range first
    # to avoid wrap-around artefacts (e.g. 256 → 0 for uint8).
    if np.issubdtype(original_dtype, np.integer):
        info    = np.iinfo(original_dtype)
        shifted = np.clip(shifted, info.min, info.max).astype(original_dtype)
    else:
        shifted = shifted.astype(original_dtype)

    # Reconstruct a PIL Image, attempting to preserve the original pixel mode
    # so that 16-bit TIFFs remain 16-bit when saved, not downgraded to 8-bit.
    try:
        shifted_image = Image.fromarray(shifted, mode=original_mode)
    except (TypeError, ValueError):
        # Fallback: let Pillow infer the mode automatically from array shape/dtype.
        shifted_image = Image.fromarray(shifted)

    return {
        "shifted_image":    shifted_image,
        "shifted_format":   original_format,
        "suggested_suffix": f".{original_suffix}",
    }



# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_calibration(image_path, show_plots=False):
    """
    Run the full star-calibration pipeline on one GONet image file.

    This function orchestrates all the steps defined above in sequence and
    returns the results to the GUI.  It also prints a short diagnostic
    summary to the console so you can monitor progress.

    Parameters
    ----------
    image_path : str or Path
        Path to the GONet image file to calibrate (JPG, TIFF, PNG, etc.).
    show_plots : bool
        If True, display matplotlib diagnostic plots (useful for development
        and debugging).  Must be False when called from the GUI because
        matplotlib’s plt.show() blocks the thread and would freeze the window.

    Returns
    -------
    dict with keys:
        best            — orientation solver result (angles, score, RMS, …)
        wcs_result      — WCS fit / zenith-centring result dict
        shifted_image   — PIL Image of the calibrated output
        shifted_format  — format string for saving (e.g. "TIFF", "JPEG")
        suggested_suffix — file extension for the save dialog (e.g. ".tiff")
    """
    # --- Load the GONet image file -----------------------------------------
    # GONetFile understands the proprietary format and gives us:
    #   .green — the green Bayer channel as a 2-D numpy array.  We use green
    #             for star detection because it has the highest resolution in
    #             a Bayer mosaic (50 % of all pixels are green).
    #   .meta  — a dict of metadata (GPS coordinates, capture date/time, …).
    go  = GONetFile.from_file(image_path)
    sub = go.green                        # shape (H, W), dtype float

    # --- Detect bright stars using a pixel-threshold mask -----------------
    # We flag any pixel more than N=5 standard deviations above the mean as
    # a potential star.  N=5 gives very few false positives on a typical
    # night-sky image while still catching all reasonably bright stars.
    N         = 5
    sub_mean  = float(np.mean(sub))
    sub_std   = float(np.std(sub))
    threshold = sub_mean + N * sub_std
    mask      = sub > threshold           # bool array: True = star candidate pixel

    # Group connected bright pixels into discrete star blobs.
    labels, num_labels = compact_search(np.array(mask, dtype=bool))

    # Find the flux-weighted centroid (x, y) of each blob.
    _, x_centroids, y_centroids = measure_sources(sub, labels, num_labels)
    img_xy = np.column_stack([x_centroids, y_centroids])  # shape (N_stars, 2)

    # --- GONet fisheye lens geometric parameters --------------------------
    # These numbers describe the specific GONet all-sky camera:
    #   cx, cy  — approximate pixel position of the optical axis (lens centre).
    #             Update these values if using a different camera setup.
    #   R_pix   — pixel radius at which the horizon (alt = 0°) appears.
    #   catalog_radius_deg — we only use the inner 60° from zenith to avoid
    #             distortion / horizon clutter at the image edge.
    cx, cy             = 1030, 760
    R_pix              = 740
    catalog_radius_deg = 60.0

    # Discard detected sources outside the sky region queried from Gaia.
    img_xy = filter_image_sources_by_radius(
        img_xy=img_xy, cx=cx, cy=cy, R_pix=R_pix,
        radius_deg=catalog_radius_deg,
    )
    if len(img_xy) == 0:
        raise RuntimeError(
            "No image centroids remain after sky-radius filtering.  "
            "Check that cx/cy/R_pix match the actual image geometry."
        )

    # --- Download catalog star positions from Gaia ------------------------
    # gmax=2.5 means we only download the ~100 brightest stars visible from
    # the camera’s location, which is more than enough to find a solid match.
    cat_alt_deg, cat_az_deg, _ = query_catalog_altaz_from_meta(
        go.meta,
        radius_deg=catalog_radius_deg,
        gmax=2.5,
        top_m=None,
    )
    if len(cat_alt_deg) == 0:
        raise RuntimeError(
            "No catalog stars found above the horizon.  "
            "Check that the GPS coordinates and observation time in the "
            "image metadata are correct."
        )

    # --- Solve for camera orientation with a grid search ------------------
    best = solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix)

    # --- Fit WCS and apply the zenith-centring shift ----------------------
    wcs_result = fit_wcs_and_center_zenith(
        sub=sub, img_xy=img_xy,
        cat_alt_deg=cat_alt_deg, cat_az_deg=cat_az_deg,
        best=best, meta=go.meta, tol_pix=25.0,
    )

    # --- Console diagnostics ----------------------------------------------
    print(f"catalog_stars={len(cat_alt_deg)}, image_sources={len(img_xy)}")
    print(f"score={best['score']}, matched={best['matched_count']}, "
          f"rms_pix={best['rms_pix']:.3f}")
    print("alpha_deg={:.3f}, beta_deg={:.3f}, gamma_deg={:.3f}".format(
        np.rad2deg(best["alpha"]),
        np.rad2deg(best["beta"]),
        np.rad2deg(best["gamma"]),
    ))
    print(f"WCS stars used = {wcs_result['n_wcs_matches']}")
    if wcs_result["wcs_fit_success"]:
        print(f"Zenith pixel (WCS): x={wcs_result['zenith_x']:.2f}, "
              f"y={wcs_result['zenith_y']:.2f}")
        print(f"Applied shift: dx={wcs_result['shift_x']:.2f}, "
              f"dy={wcs_result['shift_y']:.2f}")
    else:
        print("WCS fit failed — zenith-centring skipped.")
        print(f"Reason: {wcs_result['wcs_fit_error']}")

    # --- Optional diagnostic plots ----------------------------------------
    # Only generated when show_plots=True (e.g. during development).
    # plt.show() blocks the calling thread, which is why it must be disabled
    # when run_calibration is called from the GUI’s background worker thread.
    if show_plots:
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots()
        ax1.imshow(sub, origin="lower", cmap="gray",
                   vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
        ax1.scatter(img_xy[:, 0], img_xy[:, 1],
                    s=50, edgecolor="red",  facecolor="none", label="Detected sources")
        ax1.scatter(best["pred_xy"][:, 0], best["pred_xy"][:, 1],
                    s=50, edgecolor="blue", facecolor="none", label="Catalog predictions")
        ax1.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]],
                    s=100, marker="+", c="yellow", label="Image centre")
        if wcs_result["wcs_fit_success"]:
            ax1.scatter([wcs_result["zenith_x"]], [wcs_result["zenith_y"]],
                        s=120, marker="x", c="cyan", label="Zenith (WCS)")
            ax1.plot([wcs_result["zenith_x"], wcs_result["target_cx"]],
                     [wcs_result["zenith_y"], wcs_result["target_cy"]],
                     color="cyan", linestyle="--", linewidth=1.5, label="Applied shift")
        ax1.legend()
        ax1.set_title(f"Orientation solve — score: {best['score']} matches")
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.imshow(wcs_result["centered_sub"], origin="lower", cmap="gray",
                   vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
        ax2.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]],
                    s=120, marker="x", c="cyan", label="Zenith (centred)")
        ax2.legend()
        ax2.set_title("Shifted image — zenith at centre")
        plt.show()

    # --- Build the final shifted image from the original file -------------
    # We re-open the original file here rather than using the green-channel
    # sub-image so that the full-colour, full-bit-depth image is saved.
    shifted_result = build_shifted_image_same_format(
        image_path=image_path,
        shift_x=wcs_result["shift_x"],
        shift_y=wcs_result["shift_y"],
    )
    print("Shifted image prepared (not yet saved).")

    return {
        "best":             best,
        "wcs_result":       wcs_result,
        "shifted_image":    shifted_result["shifted_image"],
        "shifted_format":   shifted_result["shifted_format"],
        "suggested_suffix": shifted_result["suggested_suffix"],
    }