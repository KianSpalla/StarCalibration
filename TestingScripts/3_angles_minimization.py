import numpy as np
from scipy.spatial import cKDTree
from GONet_Wizard.GONet_utils import GONetFile
from pathlib import Path
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astroquery.gaia import Gaia


def compact_search(mask):
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=int)
    current_label = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue

            current_label += 1
            stack = [(row, col)]
            labels[row, col] = current_label

            while stack:
                current_row, current_col = stack.pop()
                for dr, dc in neighbors:
                    nr = current_row + dr
                    nc = current_col + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current_label
                        stack.append((nr, nc))

    return labels, current_label


def measure_sources(sub, labels, num_clusters):
    sources = []
    x_centroids = []
    y_centroids = []

    for label_id in range(1, num_clusters + 1):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
            continue

        fluxes = sub[ys, xs]
        total_flux = fluxes.sum()
        if total_flux <= 0:
            continue

        max_index = np.argmax(fluxes)
        x_peak = xs[max_index]
        y_peak = ys[max_index]
        x_centroid = (xs * fluxes).sum() / total_flux
        y_centroid = (ys * fluxes).sum() / total_flux

        source = {
            "label": label_id,
            "x_peak": x_peak,
            "y_peak": y_peak,
            "x_centroid": x_centroid,
            "y_centroid": y_centroid,
            "flux": total_flux,
            "npix": len(xs),
        }
        sources.append(source)
        x_centroids.append(x_centroid)
        y_centroids.append(y_centroid)

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
    """
    Convert (alt, az) to a unit vector in the local sky frame.
    Convention used here:
      x = east, y = north, z = up
    """
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


def rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], float)


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], float)


def orientation_matrix(alpha, beta, gamma):
    """
    Build R that maps camera vectors -> sky vectors.

    Parameters
    ----------
    alpha : float
        Rotation about camera optical axis (azimuth zero-point), radians.
    beta : float
        Tilt magnitude, radians.
    gamma : float
        Tilt direction (azimuth in sky frame), radians.

    Intern-friendly interpretation:
      - alpha spins the camera around its own axis.
      - beta and gamma tilt the camera axis away from zenith.

    TODO:
      Decide and document rotation order carefully.
      One workable choice:
        1) spin camera around its axis by alpha (in camera frame)
        2) tilt the axis by beta toward direction gamma (in sky frame)
    """
    # One reasonable construction:
    # tilt toward azimuth gamma can be done by:
    #   rotate sky frame so tilt direction aligns with +y, tilt about x or y, rotate back
    # Keep it simple: use Z(gamma) then X(beta) then Z(-gamma), combined with alpha.

    R_alpha = rot_z(alpha)  # spin about camera z (if we treat cam z as "up" initially)
    R_tilt  = rot_z(gamma) @ rot_x(beta) @ rot_z(-gamma)

    # Map camera -> sky:
    R = R_tilt @ R_alpha
    return R


# ----------------------------
# 2) Lens model (simple)
# ----------------------------
def theta_from_r(r_pix, R_pix):
    """Simple equidistant mapping: R_pix -> 90 deg."""
    # theta in radians
    return (r_pix / R_pix) * (np.pi / 2)

def r_from_theta(theta_rad, R_pix):
    """Inverse of theta_from_r."""
    return (theta_rad / (np.pi / 2)) * R_pix


# ----------------------------
# 3) Forward model: catalog Alt/Az -> predicted pixels
# ----------------------------
def predict_pixels_from_catalog(alt_deg, az_deg, cx, cy, R_pix, alpha, beta, gamma):
    alt = np.deg2rad(alt_deg)
    az  = np.deg2rad(az_deg)

    v_sky = unitvec_from_altaz(alt, az)   # (M,3)

    R = orientation_matrix(alpha, beta, gamma)   # cam -> sky
    # We need sky -> cam, so invert (rotation matrix inverse = transpose)
    v_cam = v_sky @ R.T

    # Convert cam vector -> (theta, phi)
    # theta = angle from cam +z
    z = np.clip(v_cam[:, 2], -1, 1)
    theta = np.arccos(z)
    phi = np.arctan2(v_cam[:, 1], v_cam[:, 0])

    # Convert theta -> pixel radius
    r = r_from_theta(theta, R_pix)

    # Polar -> pixel
    x = cx + r * np.cos(phi)
    y = cy + r * np.sin(phi)
    return np.stack([x, y], axis=-1)


# ----------------------------
# 4) Matching score
# ----------------------------
def match_score(img_xy, pred_xy, tol_pix=20.0):
    tree = cKDTree(img_xy)
    dists, idx = tree.query(pred_xy, k=1)
    score = np.sum(dists <= tol_pix)
    return int(score), dists, idx


# ----------------------------
# 5) Optimization loop (coarse grid + refine)
# ----------------------------
def solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix):
    """
    Solve for alpha, beta, gamma by maximizing number of matches.

    Suggested search ranges (start easy):
      alpha: 0..360 deg
      beta : 0..10 deg (camera should be close to zenith)
      gamma: 0..360 deg (only matters if beta>0)
    """
    # Coarse grid (tune for runtime)
    alpha_grid = np.deg2rad(np.arange(0, 360, 5))
    #print( alpha_grid)
    beta_grid  = np.deg2rad(np.arange(0, 11, 2))     # 0,2,4,6,8,10 deg
    gamma_grid = np.deg2rad(np.arange(0, 360, 20))   # coarse; refine later

    best = {"score": -1}

    for beta in beta_grid:
        # If beta is 0, gamma doesn’t matter — you can skip gamma loop for speed
        gamma_list = [0.0] if abs(beta) < 1e-12 else gamma_grid

        for gamma in gamma_list:
            for alpha in alpha_grid:
                pred_xy = predict_pixels_from_catalog(
                    cat_alt_deg, cat_az_deg,
                    cx, cy, R_pix,
                    alpha, beta, gamma
                )
                s, dists, idx = match_score(img_xy, pred_xy, tol_pix=25.0)
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

    # Refinement around best coarse solution
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
                s, dists, idx = match_score(img_xy, pred_xy, tol_pix=25.0)

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

def main():
    image_path = Path(r"Testing Images\256_251029_204008_1761770474.jpg")
    go = GONetFile.from_file(image_path)
    go.remove_overscan()

    N = 5
    go_ravel=np.ravel(go.green)
    sub = go.green
    sub_mean=go_ravel.mean()
    sub_std=go_ravel.std()
    threshold = sub_mean+N*sub_std
    mask = sub>threshold
    labels,num_labels=compact_search(np.array(mask, dtype=bool))
    sources,x_centroids,y_centroids=measure_sources(sub, labels, num_labels)
    img_xy = np.column_stack([x_centroids, y_centroids])

    cat_alt_deg, cat_az_deg, _ = query_catalog_altaz_from_meta(
        go.meta,
        radius_deg=60.0,
        gmax=2.5,
        top_m=None,
    )
    if len(cat_alt_deg) == 0:
        raise RuntimeError("No catalog stars available after filtering above horizon.")

    cx, cy = 1030, 760
    R_pix=740
    best=solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix)
    print(f"catalog_stars={len(cat_alt_deg)}, image_sources={len(img_xy)}")
    print(f"score={best['score']}, matched={best['matched_count']}, rms_pix={best['rms_pix']:.3f}")
    print(
        "alpha_deg={:.3f}, beta_deg={:.3f}, gamma_deg={:.3f}".format(
            np.rad2deg(best["alpha"]),
            np.rad2deg(best["beta"]),
            np.rad2deg(best["gamma"]),
        )
    )
    # Visualize results
    plt.figure()
    plt.imshow(sub, origin="lower", cmap="gray"
               ,vmin=sub_mean-2*sub_std, vmax=sub_mean+5*sub_std)
    plt.scatter(img_xy[:, 0], img_xy[:, 1], s=50,
                edgecolor="red", facecolor="none", label="Image sources")
    plt.scatter(best["pred_xy"][:, 0], best["pred_xy"][:, 1], s=50,
                edgecolor="blue", facecolor="none", label="Predicted sources")
    plt.legend()
    plt.title(f"Best score: {best['score']} matches")
    plt.show()
    

if __name__ == "__main__":
    main()