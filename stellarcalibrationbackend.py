import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import (
    label as nd_label,
    sum as nd_sum,
    center_of_mass as nd_com,
    shift as nd_shift,
)
from GONet_Wizard.GONet_utils import GONetFile
from PIL import Image
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from astroquery.gaia import Gaia


# ── Star detection ───────────────────────────────────────────────────────────

def detect_sources(img, N=5):
    """Threshold at mean + N*std, label blobs, return (n,2) xy centroid array."""
    mask = img > img.mean() + N * img.std()
    labels, n = nd_label(mask)
    if n == 0:
        return np.empty((0, 2))
    ids  = np.arange(1, n + 1)
    flux = nd_sum(img, labels, ids)
    coms = nd_com(img, labels, ids)           # list of (y, x)
    xy   = np.array([[coms[i][1], coms[i][0]] for i, f in enumerate(flux) if f > 0])
    return xy


def filter_by_radius(xy, cx, cy, R_pix, radius_deg):
    """Keep sources within the fisheye sky circle."""
    if len(xy) == 0:
        return xy
    r = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    return xy[r <= (radius_deg / 90.0) * R_pix]


# ── Gaia catalog query ───────────────────────────────────────────────────────

def query_catalog(meta, radius_deg=60.0, gmax=2.5):
    """Return (alt_deg, az_deg, gmag) arrays for Gaia stars above the horizon."""
    loc = EarthLocation(
        lat=float(meta["GPS"]["latitude"]) * u.deg,
        lon=float(meta["GPS"]["longitude"]) * u.deg,
        height=float(meta["GPS"]["altitude"]) * u.m,
    )
    t     = Time(meta["DateTime"].replace(":", "-", 2).replace(" ", "T"), scale="utc")
    frame = AltAz(obstime=t, location=loc)

    zenith = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame).icrs

    Gaia.ROW_LIMIT = 200000
    job = Gaia.launch_job_async(f"""
        SELECT ra, dec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {zenith.ra.deg}, {zenith.dec.deg}, {radius_deg})
        ) AND phot_g_mean_mag < {gmax}
    """)
    tbl   = job.get_results()
    stars = SkyCoord(ra=np.array(tbl["ra"]) * u.deg,
                     dec=np.array(tbl["dec"]) * u.deg).transform_to(frame)
    above = stars.alt.deg > 0
    return stars.alt.deg[above], stars.az.deg[above], np.array(tbl["phot_g_mean_mag"])[above]


# ── Orientation model (equisolid fisheye) ────────────────────────────────────

def _R(alpha, beta, gamma):
    """Rotation matrix: camera → sky  (azimuth α, tilt β, roll γ)."""
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)
    return Rz(gamma) @ Rx(beta) @ Rz(-gamma) @ Rz(alpha)


def predict_pixels(alt_deg, az_deg, cx, cy, R_pix, alpha, beta, gamma):
    """Project catalog alt/az into image pixels."""
    alt, az = np.deg2rad(alt_deg), np.deg2rad(az_deg)
    ca   = np.cos(alt)
    v    = np.stack([ca * np.sin(az), ca * np.cos(az), np.sin(alt)], axis=-1)
    vc   = v @ _R(alpha, beta, gamma).T
    r    = np.arccos(np.clip(vc[:, 2], -1, 1)) / (np.pi / 2) * R_pix
    phi  = np.arctan2(vc[:, 1], vc[:, 0])
    return np.stack([cx + r * np.cos(phi), cy + r * np.sin(phi)], axis=-1)


# ── Orientation solver ───────────────────────────────────────────────────────

def solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix, tol=25.0):
    """Coarse grid search + local refinement; returns best-fit orientation dict."""
    tree = cKDTree(img_xy)

    def score(a, b, g):
        pred = predict_pixels(cat_alt_deg, cat_az_deg, cx, cy, R_pix, a, b, g)
        d, idx = tree.query(pred, k=1)
        return int((d <= tol).sum()), d, idx, pred

    best = {"score": -1}

    # Coarse grid
    for b in np.deg2rad(np.arange(0, 11, 2)):
        for g in ([0.0] if b < 1e-9 else np.deg2rad(np.arange(0, 360, 20))):
            for a in np.deg2rad(np.arange(0, 360, 5)):
                s, d, idx, pred = score(a, b, g)
                if s > best["score"]:
                    best = dict(score=s, alpha=a, beta=b, gamma=g, dists=d, idx=idx, pred_xy=pred)

    # Local refinement
    for b in np.unique(np.clip(best["beta"] + np.deg2rad(np.arange(-2, 2.1, 0.2)), 0, np.deg2rad(15))):
        for g in np.unique(np.mod(best["gamma"] + np.deg2rad(np.arange(-10, 10.1, 1)), 2 * np.pi)):
            for a in np.unique(np.mod(best["alpha"] + np.deg2rad(np.arange(-5, 5.1, 0.5)), 2 * np.pi)):
                s, d, idx, pred = score(a, b, g)
                if s > best["score"]:
                    best = dict(score=s, alpha=a, beta=b, gamma=g, dists=d, idx=idx, pred_xy=pred)

    mask = best["dists"] <= tol
    best["matched_count"] = int(mask.sum())
    best["rms_pix"] = float(np.sqrt(np.mean(best["dists"][mask] ** 2))) if mask.any() else np.nan
    return best


# ── WCS fit & zenith centering ───────────────────────────────────────────────

def _wcs_fail(sub, n, error=""):
    cx, cy = (sub.shape[1] - 1) / 2, (sub.shape[0] - 1) / 2
    return dict(wcs=None, zenith_x=np.nan, zenith_y=np.nan,
                target_cx=cx, target_cy=cy, shift_x=0.0, shift_y=0.0,
                centered_sub=sub.astype(float).copy(),
                n_wcs_matches=n, wcs_fit_success=False, wcs_fit_error=error)


def fit_wcs_and_shift(sub, img_xy, alt, az, best, meta, tol=25.0):
    """Fit TAN WCS from matched star pairs, locate zenith pixel, compute shift."""
    # Deduplicate matches (closest pairing wins)
    matched_catalog_mask = best["dists"] <= tol
    matched_catalog_indices = np.where(matched_catalog_mask)[0]
    matched_image_indices = best["idx"][matched_catalog_mask]
    matched_distances_pixels = best["dists"][matched_catalog_mask]

    order   = np.argsort(matched_distances_pixels)
    seen, kept = set(), []
    for i in order:
        if matched_image_indices[i] not in seen:
            seen.add(int(matched_image_indices[i]))
            kept.append(i)

    if len(kept) < 3:
        return _wcs_fail(sub, len(kept), "Not enough matched stars for WCS fit.")

    cat_idx = matched_catalog_indices[kept];  img_idx = matched_image_indices[kept]
    px, py  = img_xy[img_idx, 0], img_xy[img_idx, 1]

    # Build AltAz frame from metadata
    loc   = EarthLocation(lat=float(meta["GPS"]["latitude"]) * u.deg,
                          lon=float(meta["GPS"]["longitude"]) * u.deg,
                          height=float(meta["GPS"]["altitude"]) * u.m)
    t     = Time(meta["DateTime"].replace(":", "-", 2).replace(" ", "T"), scale="utc")
    frame = AltAz(obstime=t, location=loc)

    sky    = SkyCoord(alt=alt[cat_idx] * u.deg, az=az[cat_idx] * u.deg, frame=frame).icrs
    zenith = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame).icrs

    # Fit WCS, try a few projection centres
    wcs = None
    for proj in ["center", zenith, sky[0], sky[len(sky) // 2]]:
        try:
            wcs = fit_wcs_from_points((px, py), sky, projection="TAN", proj_point=proj)
            break
        except Exception:
            pass

    if wcs is None:
        return _wcs_fail(sub, len(kept), "WCS fit failed for all projection centres.")

    zx, zy = zenith.to_pixel(wcs, origin=0)
    cx, cy  = (sub.shape[1] - 1) / 2, (sub.shape[0] - 1) / 2
    dx, dy  = cx - float(zx), cy - float(zy)
    shifted = nd_shift(sub.astype(float), (dy, dx), order=1,
                       mode="constant", cval=float(np.median(sub)))

    return dict(wcs=wcs, zenith_x=float(zx), zenith_y=float(zy),
                target_cx=cx, target_cy=cy, shift_x=dx, shift_y=dy,
                centered_sub=shifted, n_wcs_matches=len(kept),
                wcs_fit_success=True, wcs_fit_error="")


# ── Image shifting ───────────────────────────────────────────────────────────

def shift_image(image_path, dx, dy):
    """Shift every channel of a PIL image by (dx, dy); return dict with PIL image."""
    img   = Image.open(image_path)
    arr   = np.array(img)
    dtype = arr.dtype

    def _ch(ch):
        return nd_shift(ch.astype(float), (dy, dx), order=1,
                        mode="constant", cval=float(np.median(ch)))

    out = _ch(arr) if arr.ndim == 2 else np.stack([_ch(arr[..., c]) for c in range(arr.shape[2])], axis=-1)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        out  = np.clip(out, info.min, info.max)
    out = out.astype(dtype)

    ext = str(image_path).rsplit(".", 1)[-1].lower() if "." in str(image_path) else "png"
    return dict(shifted_image=Image.fromarray(out),
                shifted_format=img.format,
                suggested_suffix=f".{ext}")


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_calibration(image_path, show_plots=False, N=5, gmax=2.5):
    go   = GONetFile.from_file(image_path)
    sub  = np.array(go.green, dtype=float)
    meta = go.meta

    cx, cy, R_pix, radius = 1030, 760, 740, 60.0

    img_xy = detect_sources(sub, N=N)
    img_xy = filter_by_radius(img_xy, cx, cy, R_pix, radius)
    if len(img_xy) == 0:
        raise RuntimeError("No image sources found within the sky circle.")

    cat_alt, cat_az, cat_gmag = query_catalog(meta, radius_deg=radius, gmax=gmax)
    if len(cat_alt) == 0:
        raise RuntimeError("No catalog stars found above the horizon.")

    best       = solve_orientation(img_xy, cat_alt, cat_az, cx, cy, R_pix)
    wcs_result = fit_wcs_and_shift(sub, img_xy, cat_alt, cat_az, best, meta)

    print(f"sources={len(img_xy)}, catalog={len(cat_alt)}, "
          f"score={best['score']}, rms={best['rms_pix']:.2f} px")
    if wcs_result["wcs_fit_success"]:
        print(f"zenith=({wcs_result['zenith_x']:.1f}, {wcs_result['zenith_y']:.1f}), "
              f"shift=({wcs_result['shift_x']:+.1f}, {wcs_result['shift_y']:+.1f}) px")
    else:
        print(f"WCS failed: {wcs_result['wcs_fit_error']}")

    if show_plots:
        _plot(sub, img_xy, best, wcs_result)

    mask = best["dists"] <= 25.0
    matched_stars = [
        dict(alt=float(cat_alt[ci]), az=float(cat_az[ci]), gmag=float(cat_gmag[ci]),
             img_x=float(img_xy[ii, 0]), img_y=float(img_xy[ii, 1]),
             dist_px=float(best["dists"][mask][k]))
        for k, (ci, ii) in enumerate(zip(np.where(mask)[0], best["idx"][mask]))
    ]

    shifted = shift_image(image_path, wcs_result["shift_x"], wcs_result["shift_y"])
    return dict(best=best, wcs_result=wcs_result,
                matched_stars=matched_stars, sub=sub, img_xy=img_xy,
                meta=meta, **shifted)


# ── Diagnostic plots (CLI use) ───────────────────────────────────────────────

def _plot(sub, img_xy, best, wcs_result):
    import matplotlib.pyplot as plt
    m, s = sub.mean(), sub.std()
    plt.figure()
    plt.imshow(sub, origin="lower", cmap="gray", vmin=m - 2*s, vmax=m + 5*s)
    plt.scatter(img_xy[:, 0], img_xy[:, 1], s=50, edgecolor="red",  facecolor="none", label="Sources")
    plt.scatter(best["pred_xy"][:, 0], best["pred_xy"][:, 1], s=50, edgecolor="blue", facecolor="none", label="Predicted")
    plt.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]], s=100, marker="+", c="yellow", label="Target")
    if wcs_result["wcs_fit_success"]:
        plt.scatter([wcs_result["zenith_x"]], [wcs_result["zenith_y"]], s=120, marker="x", c="cyan", label="Zenith")
        plt.plot([wcs_result["zenith_x"], wcs_result["target_cx"]],
                 [wcs_result["zenith_y"], wcs_result["target_cy"]], "c--", lw=1.5, label="Shift")
    plt.legend(); plt.title(f"Score: {best['score']} matches"); plt.show()

    plt.figure()
    plt.imshow(wcs_result["centered_sub"], origin="lower", cmap="gray", vmin=m - 2*s, vmax=m + 5*s)
    plt.scatter([wcs_result["target_cx"]], [wcs_result["target_cy"]], s=120, marker="x", c="cyan", label="Centered zenith")
    plt.legend(); plt.title("Shifted image"); plt.show()