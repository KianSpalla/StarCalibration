"""
star_calibration_backend.py

Comment style in this file is intentionally change-focused.
Compared with stellarcalibrationOLDVERSION.py, this version keeps the same
pipeline but documents only concrete improvements and behavior differences.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import (
    label as nd_label,
    sum as ndimage_sum,
    center_of_mass as ndimage_com,
    shift as nd_shift,
)
from GONet_Wizard.GONet_utils import GONetFile
from PIL import Image
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from astroquery.gaia import Gaia


def compact_search(mask):
    """
    Improvement vs old version:
    - Replaces Python DFS/BFS connected-components loop with scipy.ndimage.label.
    - Same output contract (labels, count), significantly faster on large masks.
    """
    labels, num_clusters = nd_label(mask)
    return labels, num_clusters

#purpose of this functino is to take the image, mask, and number of clusters.
#the function then finds the centroids of the clusters and their total flux. 
#The centroids are found using the scipy.ndimage.center_of_mass function which 
#calculates the center of mass of the labeled stars and returns the centroids as a list of (x, y) coordinates. 
#The total flux is found using the scipy.ndimage.sum function which calculates the sum of the pixel 
#values in the subimage for each labeled star and returns the total flux for each star. 
#The function then returns a list of sources, where each source is a dictionary containing the label, x_centroid, y_centroid, and flux for each star.
def measure_sources(sub, labels, num_clusters):
    """
    Improvement vs old version:
    - Replaces per-label np.where + Python math loop with vectorized ndimage operators.
    - Removes unused peak/npix fields and keeps centroid+flux outputs used downstream.
    """
    if num_clusters == 0:
        return [], [], []

    #
    label_ids = np.arange(1, num_clusters + 1)
    #
    total_fluxes = ndimage_sum(sub, labels, index=label_ids)
    centers = ndimage_com(sub, labels, index=label_ids)

    sources = []
    x_centroids = []
    y_centroids = []

    for i, label_id in enumerate(label_ids):
        tf = float(total_fluxes[i])
        if tf <= 0:
            continue

        y_c, x_c = centers[i]

        sources.append(
            {
                "label": int(label_id),
                "x_centroid": float(x_c),
                "y_centroid": float(y_c),
                "flux": tf,
            }
        )
        x_centroids.append(float(x_c))
        y_centroids.append(float(y_c))

    return sources, x_centroids, y_centroids


def query_catalog_altaz_from_meta(meta, radius_deg=60.0, gmax=2.5, top_m=None):
    """
    Same core behavior as old version; comments reduced to implementation deltas only.
    """
    lat_deg = float(meta["GPS"]["latitude"])
    lon_deg = float(meta["GPS"]["longitude"])
    alt_m = float(meta["GPS"]["altitude"])
    ut_iso = meta["DateTime"].replace(":", "-", 2).replace(" ", "T")

    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    obstime = Time(ut_iso, scale="utc")

    zenith_altaz = SkyCoord(
        alt=90 * u.deg,
        az=0 * u.deg,
        frame=AltAz(obstime=obstime, location=location),
    )
    zenith_icrs = zenith_altaz.icrs

    Gaia.ROW_LIMIT = 200000
    ra0 = zenith_icrs.ra.deg
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

    job = Gaia.launch_job_async(query)
    tbl = job.get_results()

    stars_icrs = SkyCoord(ra=np.array(tbl["ra"]) * u.deg, dec=np.array(tbl["dec"]) * u.deg, frame="icrs")
    stars_altaz = stars_icrs.transform_to(AltAz(obstime=obstime, location=location))

    alt = stars_altaz.alt.deg
    az = stars_altaz.az.deg
    gmag = np.array(tbl["phot_g_mean_mag"])

    above = alt > 0
    alt, az, gmag = alt[above], az[above], gmag[above]

    if top_m is not None and len(gmag) > top_m:
        idx = np.argsort(gmag)[:top_m]
        alt, az, gmag = alt[idx], az[idx], gmag[idx]

    return alt, az, gmag


def unitvec_from_altaz(alt_rad, az_rad):
    ca = np.cos(alt_rad)
    x = ca * np.sin(az_rad)
    y = ca * np.cos(az_rad)
    z = np.sin(alt_rad)
    return np.stack([x, y, z], axis=-1)


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def orientation_matrix(alpha, beta, gamma):
    R_alpha = rot_z(alpha)
    R_tilt = rot_z(gamma) @ rot_x(beta) @ rot_z(-gamma)
    return R_tilt @ R_alpha


def r_from_theta(theta_rad, R_pix):
    return (theta_rad / (np.pi / 2)) * R_pix


def filter_image_sources_by_radius(img_xy, cx, cy, R_pix, radius_deg):
    if len(img_xy) == 0:
        return img_xy

    max_r_pix = (float(radius_deg) / 90.0) * float(R_pix)
    dx = img_xy[:, 0] - float(cx)
    dy = img_xy[:, 1] - float(cy)
    rr = np.sqrt(dx * dx + dy * dy)
    keep = rr <= max_r_pix
    return img_xy[keep]


def predict_pixels_from_catalog(alt_deg, az_deg, cx, cy, R_pix, alpha, beta, gamma):
    alt = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)

    v_sky = unitvec_from_altaz(alt, az)
    R = orientation_matrix(alpha, beta, gamma)
    v_cam = v_sky @ R.T

    z = np.clip(v_cam[:, 2], -1, 1)
    theta = np.arccos(z)
    phi = np.arctan2(v_cam[:, 1], v_cam[:, 0])

    r = r_from_theta(theta, R_pix)
    x = cx + r * np.cos(phi)
    y = cy + r * np.sin(phi)
    return np.stack([x, y], axis=-1)


def match_score(img_tree, pred_xy, tol_pix=20.0):
    """
    Improvement vs old version:
    - Old function built cKDTree on every call.
    - New function receives a prebuilt tree, removing thousands of rebuilds
      during grid search.
    """
    dists, idx = img_tree.query(pred_xy, k=1)
    score = np.sum(dists <= tol_pix)
    return int(score), dists, idx


def solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix):
    """
    Improvement vs old version:
    - Builds cKDTree once and reuses it via match_score(img_tree, ...).
    - Keeps same coarse+refine search behavior and output fields.
    """
    img_tree = cKDTree(img_xy)

    alpha_grid = np.deg2rad(np.arange(0, 360, 5))
    beta_grid = np.deg2rad(np.arange(0, 11, 2))
    gamma_grid = np.deg2rad(np.arange(0, 360, 20))

    best = {"score": -1}

    for beta in beta_grid:
        gamma_list = [0.0] if abs(beta) < 1e-12 else gamma_grid
        for gamma in gamma_list:
            for alpha in alpha_grid:
                pred_xy = predict_pixels_from_catalog(cat_alt_deg, cat_az_deg, cx, cy, R_pix, alpha, beta, gamma)
                s, dists, idx = match_score(img_tree, pred_xy, tol_pix=25.0)
                if s > best["score"]:
                    best = {
                        "score": s,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "dists": dists,
                        "idx": idx,
                        "pred_xy": pred_xy,
                    }

    alpha_refine = best["alpha"] + np.deg2rad(np.arange(-5.0, 5.0 + 1e-12, 0.5))
    beta_refine = best["beta"] + np.deg2rad(np.arange(-2.0, 2.0 + 1e-12, 0.2))
    gamma_refine = best["gamma"] + np.deg2rad(np.arange(-10.0, 10.0 + 1e-12, 1.0))

    alpha_refine = np.mod(alpha_refine, 2 * np.pi)
    gamma_refine = np.mod(gamma_refine, 2 * np.pi)
    beta_refine = np.clip(beta_refine, 0.0, np.deg2rad(15.0))

    for beta in np.unique(beta_refine):
        gamma_list = [best["gamma"]] if abs(beta) < 1e-12 else np.unique(gamma_refine)
        for gamma in gamma_list:
            for alpha in np.unique(alpha_refine):
                pred_xy = predict_pixels_from_catalog(cat_alt_deg, cat_az_deg, cx, cy, R_pix, alpha, beta, gamma)
                s, dists, idx = match_score(img_tree, pred_xy, tol_pix=25.0)
                if s > best["score"]:
                    best = {
                        "score": s,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "dists": dists,
                        "idx": idx,
                        "pred_xy": pred_xy,
                    }

    matched_mask = best["dists"] <= 25.0
    matched_count = int(np.sum(matched_mask))
    if matched_count > 0:
        best["rms_pix"] = float(np.sqrt(np.mean(best["dists"][matched_mask] ** 2)))
    else:
        best["rms_pix"] = np.nan
    best["matched_count"] = matched_count
    return best


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
    Improvements vs old version:
    - Adds shared _failure(...) helper so all failure returns are consistent.
    - Keeps deduplication/projection-center retry logic, but failure messages are
      normalized and easier to consume by the GUI.
    """

    def _failure(msg, n=0):
        tcx = (sub.shape[1] - 1) / 2.0
        tcy = (sub.shape[0] - 1) / 2.0
        return {
            "wcs": None,
            "zenith_x": np.nan,
            "zenith_y": np.nan,
            "target_cx": float(tcx),
            "target_cy": float(tcy),
            "shift_x": 0.0,
            "shift_y": 0.0,
            "centered_sub": sub.astype(float).copy(),
            "n_wcs_matches": n,
            "wcs_fit_success": False,
            "wcs_fit_error": msg,
        }

    matched_catalog_mask = best["dists"] <= tol_pix
    matched_catalog_indices = np.where(matched_catalog_mask)[0]
    matched_image_indices = best["idx"][matched_catalog_mask]
    matched_distances = best["dists"][matched_catalog_mask]

    sorted_order = np.argsort(matched_distances)
    used_image_set = set()
    kept_positions = []
    for pos in sorted_order:
        img_idx = int(matched_image_indices[pos])
        if img_idx in used_image_set:
            continue
        used_image_set.add(img_idx)
        kept_positions.append(pos)

    min_wcs_matches = max(3, int(min_wcs_matches))
    if len(kept_positions) < min_wcs_matches:
        return _failure(
            f"Not enough unique matches for WCS fit ({len(kept_positions)} < {min_wcs_matches}).",
            n=len(kept_positions),
        )

    kept_positions = np.array(kept_positions, dtype=int)
    matched_catalog_indices = matched_catalog_indices[kept_positions]
    matched_image_indices = matched_image_indices[kept_positions]

    matched_pixel_x = img_xy[matched_image_indices, 0]
    matched_pixel_y = img_xy[matched_image_indices, 1]

    lat_deg = float(meta["GPS"]["latitude"])
    lon_deg = float(meta["GPS"]["longitude"])
    alt_m = float(meta["GPS"]["altitude"])
    ut_iso = meta["DateTime"].replace(":", "-", 2).replace(" ", "T")

    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    obstime = Time(ut_iso, scale="utc")

    matched_altaz = SkyCoord(
        alt=np.array(cat_alt_deg)[matched_catalog_indices] * u.deg,
        az=np.array(cat_az_deg)[matched_catalog_indices] * u.deg,
        frame=AltAz(obstime=obstime, location=location),
    )
    matched_icrs = matched_altaz.icrs

    zenith_altaz = SkyCoord(alt=90.0 * u.deg, az=0.0 * u.deg, frame=AltAz(obstime=obstime, location=location))
    zenith_icrs = zenith_altaz.icrs

    proj_center_candidates = [
        "center",
        zenith_icrs,
        matched_icrs[0],
        matched_icrs[len(matched_icrs) // 2],
    ]

    fitted_wcs = None
    last_wcs_error = None
    for proj_center in proj_center_candidates:
        try:
            fitted_wcs = fit_wcs_from_points(
                (matched_pixel_x, matched_pixel_y),
                matched_icrs,
                projection="TAN",
                proj_point=proj_center,
            )
            break
        except Exception as e:
            last_wcs_error = e

    if fitted_wcs is None:
        return _failure(str(last_wcs_error), n=len(kept_positions))

    zenith_pixel_x, zenith_pixel_y = zenith_icrs.to_pixel(fitted_wcs, origin=0)

    target_cx = (sub.shape[1] - 1) / 2.0
    target_cy = (sub.shape[0] - 1) / 2.0
    shift_x = target_cx - float(zenith_pixel_x)
    shift_y = target_cy - float(zenith_pixel_y)

    centered_sub = nd_shift(
        sub.astype(float),
        shift=(shift_y, shift_x),
        order=1,
        mode="constant",
        cval=float(np.median(sub)),
    )

    return {
        "wcs": fitted_wcs,
        "zenith_x": float(zenith_pixel_x),
        "zenith_y": float(zenith_pixel_y),
        "target_cx": float(target_cx),
        "target_cy": float(target_cy),
        "shift_x": float(shift_x),
        "shift_y": float(shift_y),
        "centered_sub": centered_sub,
        "n_wcs_matches": int(len(kept_positions)),
        "wcs_fit_success": True,
        "wcs_fit_error": "",
    }


def build_shifted_image_same_format(image_path, shift_x, shift_y):
    """
    Improvements vs old version:
    - Preserves original PIL mode when possible.
    - Adds safe fallback for missing format (defaults to PNG).
    - Keeps dtype clipping behavior for integer images.
    """
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
            f"Unsupported image array shape {image_array.shape}.  "
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


def run_calibration(image_path, show_plots=False, N = 5, gmax = 2.5):
    """
    Improvement vs old version:
    - Adds show_plots flag to prevent blocking GUI workflows.
    - Maintains identical calibration/return contract.
    """
    go = GONetFile.from_file(image_path)
    sub = go.green

    sub_mean = float(np.mean(sub))
    sub_std = float(np.std(sub))
    threshold = sub_mean + N * sub_std
    mask = sub > threshold

    labels, num_labels = compact_search(np.array(mask, dtype=bool))
    _, x_centroids, y_centroids = measure_sources(sub, labels, num_labels)
    img_xy = np.column_stack([x_centroids, y_centroids])

    cx, cy = 1030, 760
    R_pix = 740
    catalog_radius_deg = 60.0

    img_xy = filter_image_sources_by_radius(
        img_xy=img_xy,
        cx=cx,
        cy=cy,
        R_pix=R_pix,
        radius_deg=catalog_radius_deg,
    )
    if len(img_xy) == 0:
        raise RuntimeError(
            "No image centroids remain after sky-radius filtering.  "
            "Check that cx/cy/R_pix match the actual image geometry."
        )

    cat_alt_deg, cat_az_deg, _ = query_catalog_altaz_from_meta(
        go.meta,
        radius_deg=catalog_radius_deg,
        gmax=gmax,
        top_m=None,
    )
    if len(cat_alt_deg) == 0:
        raise RuntimeError(
            "No catalog stars found above the horizon.  "
            "Check that the GPS coordinates and observation time in the "
            "image metadata are correct."
        )

    best = solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix)

    wcs_result = fit_wcs_and_center_zenith(
        sub=sub,
        img_xy=img_xy,
        cat_alt_deg=cat_alt_deg,
        cat_az_deg=cat_az_deg,
        best=best,
        meta=go.meta,
        tol_pix=25.0,
    )

    print(f"catalog_stars={len(cat_alt_deg)}, image_sources={len(img_xy)}")
    print(f"score={best['score']}, matched={best['matched_count']}, rms_pix={best['rms_pix']:.3f}")
    print(
        "alpha_deg={:.3f}, beta_deg={:.3f}, gamma_deg={:.3f}".format(
            np.rad2deg(best["alpha"]),
            np.rad2deg(best["beta"]),
            np.rad2deg(best["gamma"]),
        )
    )
    print(f"WCS stars used = {wcs_result['n_wcs_matches']}")
    if wcs_result["wcs_fit_success"]:
        print(f"Zenith pixel (WCS): x={wcs_result['zenith_x']:.2f}, y={wcs_result['zenith_y']:.2f}")
        print(f"Applied shift: dx={wcs_result['shift_x']:.2f}, dy={wcs_result['shift_y']:.2f}")
    else:
        print("WCS fit failed — zenith-centring skipped.")
        print(f"Reason: {wcs_result['wcs_fit_error']}")

    if show_plots:
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots()
        ax1.imshow(sub, origin="lower", cmap="gray", vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
        ax1.scatter(img_xy[:, 0], img_xy[:, 1], s=50, edgecolor="red", facecolor="none", label="Detected sources")
        ax1.scatter(best["pred_xy"][:, 0], best["pred_xy"][:, 1], s=50, edgecolor="blue", facecolor="none", label="Catalog predictions")
        ax1.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]], s=100, marker="+", c="yellow", label="Image centre")
        if wcs_result["wcs_fit_success"]:
            ax1.scatter([wcs_result["zenith_x"]], [wcs_result["zenith_y"]], s=120, marker="x", c="cyan", label="Zenith (WCS)")
            ax1.plot(
                [wcs_result["zenith_x"], wcs_result["target_cx"]],
                [wcs_result["zenith_y"], wcs_result["target_cy"]],
                color="cyan",
                linestyle="--",
                linewidth=1.5,
                label="Applied shift",
            )
        ax1.legend()
        ax1.set_title(f"Orientation solve — score: {best['score']} matches")
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.imshow(
            wcs_result["centered_sub"],
            origin="lower",
            cmap="gray",
            vmin=sub_mean - 2 * sub_std,
            vmax=sub_mean + 5 * sub_std,
        )
        ax2.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]], s=120, marker="x", c="cyan", label="Zenith (centred)")
        ax2.legend()
        ax2.set_title("Shifted image — zenith at centre")
        plt.show()

    shifted_result = build_shifted_image_same_format(
        image_path=image_path,
        shift_x=wcs_result["shift_x"],
        shift_y=wcs_result["shift_y"],
    )
    print("Shifted image prepared (not yet saved).")

    return {
        "best": best,
        "wcs_result": wcs_result,
        "shifted_image": shifted_result["shifted_image"],
        "shifted_format": shifted_result["shifted_format"],
        "suggested_suffix": shifted_result["suggested_suffix"],
    }

if __name__ == "__main__":
    run_calibration(
        r"Testing Images\256_251029_204008_1761770474.jpg",
        show_plots=True,
        N=5,
        gmax=2.5,
    )
