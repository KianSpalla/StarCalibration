import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import (
    label      as nd_label,
    sum        as ndimage_sum,
    center_of_mass as ndimage_com,
    shift      as nd_shift,
)
from GONet_Wizard.GONet_utils import GONetFile
from PIL import Image
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from astroquery.gaia import Gaia

# Replaced pure-Python BFS with scipy.ndimage.label (C implementation; ~10-100x faster).
def compact_search(mask):
    labels, num_clusters = nd_label(mask)
    return labels, num_clusters


# Replaced per-cluster Python loops with vectorized scipy.ndimage bulk functions.
def measure_sources(sub, labels, num_clusters):
    if num_clusters == 0:
        return [], [], []

    label_ids = np.arange(1, num_clusters + 1)
    total_fluxes = ndimage_sum(sub, labels, index=label_ids)    # sum per label
    centers      = ndimage_com(sub, labels, index=label_ids)    # (y,x) per label

    sources      = []
    x_centroids  = []
    y_centroids  = []

    for i, label_id in enumerate(label_ids):
        tf = float(total_fluxes[i])
        if tf <= 0:
            continue
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


def query_catalog_altaz_from_meta(meta, radius_deg=60.0, gmax=2.5, top_m=None):
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
    alt = alt[above]
    az = az[above]
    gmag = gmag[above]

    if top_m is not None and len(gmag) > top_m:
        idx = np.argsort(gmag)[:top_m]
        alt = alt[idx]
        az = az[idx]
        gmag = gmag[idx]

    return alt, az, gmag


# ----------------------------
# 1) Small math helpers
# ----------------------------
def unitvec_from_altaz(alt_rad, az_rad):
    ca = np.cos(alt_rad)
    x = ca * np.sin(az_rad)   # east
    y = ca * np.cos(az_rad)   # north
    z = np.sin(alt_rad)       # up
    return np.stack([x, y, z], axis=-1)


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], float)


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], float)


def orientation_matrix(alpha, beta, gamma):
    """
    Build R that maps camera vectors -> sky vectors.
    """
    R_alpha = rot_z(alpha)
    R_tilt  = rot_z(gamma) @ rot_x(beta) @ rot_z(-gamma)
    return R_tilt @ R_alpha


# ----------------------------
# 2) Lens model (simple)
# ----------------------------
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


# ----------------------------
# 3) Forward model: catalog Alt/Az -> predicted pixels
# ----------------------------
def predict_pixels_from_catalog(alt_deg, az_deg, cx, cy, R_pix, alpha, beta, gamma):
    alt = np.deg2rad(alt_deg)
    az  = np.deg2rad(az_deg)

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


# ----------------------------
# 4) Matching score
# ----------------------------
def match_score(img_tree, pred_xy, tol_pix=20.0):
    """img_tree is a pre-built cKDTree of image detections."""
    dists, idx = img_tree.query(pred_xy, k=1)
    score = np.sum(dists <= tol_pix)
    return int(score), dists, idx


# ----------------------------
# 5) Optimization loop (coarse grid + refine)
# ----------------------------
def solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix):
    # Build the KDTree ONCE here and pass it to every match_score call.
    # Previously a new tree was rebuilt on every iteration (~7 700+ times).
    img_tree = cKDTree(img_xy)

    alpha_grid = np.deg2rad(np.arange(0, 360, 5))
    beta_grid  = np.deg2rad(np.arange(0, 11, 2))
    gamma_grid = np.deg2rad(np.arange(0, 360, 20))

    best = {"score": -1}

    for beta in beta_grid:
        gamma_list = [0.0] if abs(beta) < 1e-12 else gamma_grid
        for gamma in gamma_list:
            for alpha in alpha_grid:
                pred_xy = predict_pixels_from_catalog(
                    cat_alt_deg, cat_az_deg,
                    cx, cy, R_pix,
                    alpha, beta, gamma
                )
                s, dists, idx = match_score(img_tree, pred_xy, tol_pix=25.0)
                if s > best["score"]:
                    best = {
                        "score": s,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "dists": dists,
                        "idx": idx,
                        "pred_xy": pred_xy
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
                pred_xy = predict_pixels_from_catalog(
                    cat_alt_deg, cat_az_deg,
                    cx, cy, R_pix,
                    alpha, beta, gamma
                )
                s, dists, idx = match_score(img_tree, pred_xy, tol_pix=25.0)

                if s > best["score"]:
                    best = {
                        "score": s,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "dists": dists,
                        "idx": idx,
                        "pred_xy": pred_xy
                    }

    matched_mask = best["dists"] <= 25.0
    matched_count = int(np.sum(matched_mask))
    if matched_count > 0:
        best["rms_pix"] = float(np.sqrt(np.mean(best["dists"][matched_mask] ** 2)))
    else:
        best["rms_pix"] = np.nan
    best["matched_count"] = matched_count

    return best


def fit_wcs_and_center_zenith(sub, img_xy, cat_alt_deg, cat_az_deg, best, meta,tol_pix=25.0, min_wcs_matches=3,):
    # Keep only catalog stars that matched within the tolerance.
    matched_catalog_mask = best["dists"] <= tol_pix
    matched_catalog_indices = np.where(matched_catalog_mask)[0]
    matched_image_indices = best["idx"][matched_catalog_mask]
    matched_distances_pixels = best["dists"][matched_catalog_mask]

    # Deduplicate matches: if multiple catalog stars map to the same image star,
    # keep the closest one first for a cleaner WCS fit.
    sorted_match_order = np.argsort(matched_distances_pixels)
    used_image_index_set = set()
    kept_match_positions = []
    for match_position in sorted_match_order:
        image_index = int(matched_image_indices[match_position])
        if image_index in used_image_index_set:
            continue
        used_image_index_set.add(image_index)
        kept_match_positions.append(match_position)

    # Final matched pixel points used for fitting.
    min_wcs_matches = max(3, int(min_wcs_matches))
    if len(kept_match_positions) < min_wcs_matches:
        target_center_x = (sub.shape[1] - 1) / 2.0
        target_center_y = (sub.shape[0] - 1) / 2.0
        return {
            "wcs": None,
            "zenith_x": np.nan,
            "zenith_y": np.nan,
            "target_cx": float(target_center_x),
            "target_cy": float(target_center_y),
            "shift_x": 0.0,
            "shift_y": 0.0,
            "centered_sub": sub.astype(float).copy(),
            "n_wcs_matches": int(len(kept_match_positions)),
            "wcs_fit_success": False,
            "wcs_fit_error": (
                f"Not enough unique image matches after deduplication for WCS fit "
                f"({len(kept_match_positions)} < {min_wcs_matches})."
            ),
        }

    kept_match_positions = np.array(kept_match_positions, dtype=int)
    matched_catalog_indices = matched_catalog_indices[kept_match_positions]
    matched_image_indices = matched_image_indices[kept_match_positions]

    matched_pixel_x = img_xy[matched_image_indices, 0]
    matched_pixel_y = img_xy[matched_image_indices, 1]

    # Rebuild observation context from metadata (location + UTC time).
    latitude_degrees = float(meta["GPS"]["latitude"])
    longitude_degrees = float(meta["GPS"]["longitude"])
    altitude_meters = float(meta["GPS"]["altitude"])
    observation_time_iso = meta["DateTime"].replace(":", "-", 2).replace(" ", "T")
    observer_location = EarthLocation(
        lat=latitude_degrees * u.deg,
        lon=longitude_degrees * u.deg,
        height=altitude_meters * u.m,
    )
    observation_time = Time(observation_time_iso, scale="utc")

    # Build matched sky coordinates in Alt/Az, then convert to ICRS for WCS fitting.
    matched_altitude_azimuth_coordinates = SkyCoord(
        alt=np.array(cat_alt_deg)[matched_catalog_indices] * u.deg,
        az=np.array(cat_az_deg)[matched_catalog_indices] * u.deg,
        frame=AltAz(obstime=observation_time, location=observer_location),
    )
    matched_icrs_coordinates = matched_altitude_azimuth_coordinates.icrs

    # Compute where zenith falls in pixel coordinates according to this WCS.
    zenith_altitude_azimuth_coordinate = SkyCoord(
        alt=90.0 * u.deg,
        az=0.0 * u.deg,
        frame=AltAz(obstime=observation_time, location=observer_location),
    )
    zenith_icrs_coordinate = zenith_altitude_azimuth_coordinate.icrs

    # Fit a TAN-projection WCS; try multiple projection centers to avoid
    # occasional optimizer bound failures on some images.
    projection_center_candidates = [
        "center",
        zenith_icrs_coordinate,
        matched_icrs_coordinates[0],
        matched_icrs_coordinates[len(matched_icrs_coordinates) // 2],
    ]
    fitted_world_coordinate_system = None
    last_wcs_error = None
    for projection_center in projection_center_candidates:
        try:
            fitted_world_coordinate_system = fit_wcs_from_points(
                (matched_pixel_x, matched_pixel_y),
                matched_icrs_coordinates,
                projection="TAN",
                proj_point=projection_center,
            )
            break
        except Exception as fit_error:
            last_wcs_error = fit_error

    if fitted_world_coordinate_system is None:
        target_center_x = (sub.shape[1] - 1) / 2.0
        target_center_y = (sub.shape[0] - 1) / 2.0
        return {
            "wcs": None,
            "zenith_x": np.nan,
            "zenith_y": np.nan,
            "target_cx": float(target_center_x),
            "target_cy": float(target_center_y),
            "shift_x": 0.0,
            "shift_y": 0.0,
            "centered_sub": sub.astype(float).copy(),
            "n_wcs_matches": int(len(kept_match_positions)),
            "wcs_fit_success": False,
            "wcs_fit_error": str(last_wcs_error),
        }

    zenith_pixel_x, zenith_pixel_y = zenith_icrs_coordinate.to_pixel(fitted_world_coordinate_system, origin=0)

    # Desired image center target for zenith after correction.
    target_center_x = (sub.shape[1] - 1) / 2.0
    target_center_y = (sub.shape[0] - 1) / 2.0

    # Translation to move zenith from its original pixel to the target center.
    shift_x_pixels = target_center_x - float(zenith_pixel_x)
    shift_y_pixels = target_center_y - float(zenith_pixel_y)

    # Shift image with linear interpolation. Fill exposed edges with median background.
    centered_sub = nd_shift(
        sub.astype(float),
        shift=(shift_y_pixels, shift_x_pixels),
        order=1,
        mode="constant",
        cval=float(np.median(sub)),
    )

    return {
        "wcs": fitted_world_coordinate_system,
        "zenith_x": float(zenith_pixel_x),
        "zenith_y": float(zenith_pixel_y),
        "target_cx": float(target_center_x),
        "target_cy": float(target_center_y),
        "shift_x": float(shift_x_pixels),
        "shift_y": float(shift_y_pixels),
        "centered_sub": centered_sub,
        "n_wcs_matches": int(len(kept_match_positions)),
        "wcs_fit_success": True,
        "wcs_fit_error": "",
    }


def build_shifted_image_same_format(image_path, shift_x, shift_y):
    pil_image = Image.open(image_path)
    original_format = pil_image.format
    original_suffix = str(image_path).split(".")[-1].lower() if "." in str(image_path) else "png"
    image_array = np.array(pil_image)
    original_dtype = image_array.dtype

    if image_array.ndim == 2:
        shifted = nd_shift(
            image_array.astype(float),
            shift=(float(shift_y), float(shift_x)),
            order=1,
            mode="constant",
            cval=float(np.median(image_array)),
        )
    elif image_array.ndim == 3:
        shifted_channels = []
        for channel_index in range(image_array.shape[2]):
            channel = image_array[..., channel_index]
            shifted_channel = nd_shift(
                channel.astype(float),
                shift=(float(shift_y), float(shift_x)),
                order=1,
                mode="constant",
                cval=float(np.median(channel)),
            )
            shifted_channels.append(shifted_channel)
        shifted = np.stack(shifted_channels, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape for shifting: {image_array.shape}")

    if np.issubdtype(original_dtype, np.integer):
        dtype_info = np.iinfo(original_dtype)
        shifted = np.clip(shifted, dtype_info.min, dtype_info.max).astype(original_dtype)
    else:
        shifted = shifted.astype(original_dtype)

    shifted_image = Image.fromarray(shifted)
    return {
        "shifted_image": shifted_image,
        "shifted_format": original_format,
        "suggested_suffix": f".{original_suffix}",
    }


def run_calibration(image_path, show_plots=False):
    go = GONetFile.from_file(image_path)
    N = 5
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
        raise RuntimeError("No image centroids left after sky-radius filtering.")

    cat_alt_deg, cat_az_deg, _ = query_catalog_altaz_from_meta(
        go.meta,
        radius_deg=catalog_radius_deg,
        gmax=2.5,
        top_m=None,
    )
    if len(cat_alt_deg) == 0:
        raise RuntimeError("No catalog stars available after filtering above horizon.")

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
    print(f"WCS stars used={wcs_result['n_wcs_matches']}")
    if wcs_result["wcs_fit_success"]:
        print(f"Zenith pixel from WCS: x={wcs_result['zenith_x']:.2f}, y={wcs_result['zenith_y']:.2f}")
        print(f"Applied shift to center zenith: dx={wcs_result['shift_x']:.2f}, dy={wcs_result['shift_y']:.2f}")
    else:
        print("WCS fit failed for this image; skipping zenith-centering shift.")
        print(f"WCS fit error: {wcs_result['wcs_fit_error']}")

    if show_plots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(sub, origin="lower", cmap="gray", vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
        plt.scatter(img_xy[:, 0], img_xy[:, 1], s=50, edgecolor="red", facecolor="none", label="Image sources")
        plt.scatter(best["pred_xy"][:, 0], best["pred_xy"][:, 1], s=50, edgecolor="blue", facecolor="none", label="Predicted sources")
        plt.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]], s=100, marker="+", c="yellow", label="Target center")
        if wcs_result["wcs_fit_success"]:
            plt.scatter([wcs_result["zenith_x"]], [wcs_result["zenith_y"]], s=120, marker="x", c="cyan", label="Zenith original (WCS)")
            plt.plot(
                [wcs_result["zenith_x"], wcs_result["target_cx"]],
                [wcs_result["zenith_y"], wcs_result["target_cy"]],
                color="cyan",
                linestyle="--",
                linewidth=1.5,
                label="Applied shift",
            )
        plt.legend()
        plt.title(f"Best score: {best['score']} matches")
        plt.show()
        plt.figure()
        plt.imshow(wcs_result["centered_sub"], origin="lower", cmap="gray", vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
        plt.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]], s=120, marker="x", c="cyan", label="Centered zenith target")
        plt.legend()
        plt.title("Image shifted so zenith is centered")
        plt.show()

    shifted_image_result = build_shifted_image_same_format(
        image_path=image_path,
        shift_x=wcs_result["shift_x"],
        shift_y=wcs_result["shift_y"],
    )
    print("Shifted image prepared (not saved yet).")

    return {
        "best": best,
        "wcs_result": wcs_result,
        "shifted_image": shifted_image_result["shifted_image"],
        "shifted_format": shifted_image_result["shifted_format"],
        "suggested_suffix": shifted_image_result["suggested_suffix"],
    }