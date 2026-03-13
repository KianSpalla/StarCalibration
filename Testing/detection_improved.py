import numpy as np
from scipy.ndimage import (
    label as nd_label,
    sum as ndimage_sum,
    center_of_mass as ndimage_com,
)


def adaptive_threshold_mask(img, tile_size=128, n_sigma=5.0):
    """
    Compute a spatially-adaptive detection mask by estimating the background
    and noise level in non-overlapping tiles. Uses the median and NMAD
    (1.4826 * median absolute deviation) per tile, which are robust to the
    bright stars within each tile.

    Returns
    -------
    mask       : bool array, True where a pixel is a candidate source
    background : float array, per-pixel background estimate
    noise      : float array, per-pixel noise (NMAD) estimate
    """
    h, w = img.shape
    background = np.empty_like(img, dtype=float)
    noise = np.empty_like(img, dtype=float)

    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            tile = img[y0 : y0 + tile_size, x0 : x0 + tile_size].astype(float)
            med = float(np.median(tile))
            nmad = 1.4826 * float(np.median(np.abs(tile - med)))
            background[y0 : y0 + tile_size, x0 : x0 + tile_size] = med
            # Guard against pathologically flat tiles
            noise[y0 : y0 + tile_size, x0 : x0 + tile_size] = max(nmad, 1e-6)

    mask = img > background + n_sigma * noise
    return mask, background, noise


def compact_search(mask):
    labels, num_clusters = nd_label(mask)
    return labels, num_clusters


def measure_sources(
    img,
    labels,
    num_clusters,
    background=None,
    min_pixels=3,
    max_pixels=200,
    min_roundness=0.3,
    top_n=None,
):
    """
    Measure detected blobs and return filtered, ranked source lists.

    Improvements over the original:
    - Background-subtracted centroids and fluxes (eliminates sky pedestal bias).
    - Pixel-count filter (min_pixels / max_pixels) removes hot pixels and
      extended non-stellar sources.
    - Roundness filter rejects satellite trails and cosmic rays.
      Roundness = minor_axis / major_axis in [0, 1]; 1.0 is perfectly circular.
    - Sources are sorted by flux (brightest first) and optionally capped at
      top_n, so the solver works with the most star-like detections.

    Parameters
    ----------
    img           : 2-D array, raw image
    labels        : 2-D int array from compact_search
    num_clusters  : int, number of blobs
    background    : 2-D float array or None (falls back to global median)
    min_pixels    : minimum blob area in pixels
    max_pixels    : maximum blob area in pixels
    min_roundness : minimum axis ratio [0, 1] to accept a blob
    top_n         : if set, return only the brightest N sources

    Returns
    -------
    sources      : list of dicts with keys x_centroid, y_centroid, flux,
                   npix, roundness
    x_centroids  : list of float
    y_centroids  : list of float
    """
    if num_clusters == 0:
        return [], [], []

    if background is None:
        background = np.full(img.shape, float(np.median(img)))

    img_sub = img.astype(float) - background
    label_ids = np.arange(1, num_clusters + 1)

    # Precompute per-blob statistics efficiently with ndimage
    ones = np.ones_like(img, dtype=float)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)

    pixel_counts = ndimage_sum(ones, labels, index=label_ids)
    total_fluxes = ndimage_sum(img_sub, labels, index=label_ids)
    centers = ndimage_com(img_sub, labels, index=label_ids)

    # Moments needed for 2-D covariance (roundness)
    sum_x = ndimage_sum(xx, labels, index=label_ids)
    sum_y = ndimage_sum(yy, labels, index=label_ids)
    sum_xx = ndimage_sum(xx * xx, labels, index=label_ids)
    sum_yy = ndimage_sum(yy * yy, labels, index=label_ids)
    sum_xy = ndimage_sum(xx * yy, labels, index=label_ids)

    sources = []

    for i, label_id in enumerate(label_ids):
        npix = int(pixel_counts[i])
        if npix < min_pixels or npix > max_pixels:
            continue

        tf = float(total_fluxes[i])
        if tf <= 0:
            continue

        # Roundness via 2-D covariance eigenvalues
        if npix >= 2:
            n = float(npix)
            mx = sum_x[i] / n
            my = sum_y[i] / n
            var_x = sum_xx[i] / n - mx * mx
            var_y = sum_yy[i] / n - my * my
            cov_xy = sum_xy[i] / n - mx * my
            trace = var_x + var_y
            det = var_x * var_y - cov_xy * cov_xy
            disc = max(0.0, (trace * trace) / 4.0 - det)
            lam_major = trace / 2.0 + np.sqrt(disc)
            lam_minor = max(0.0, trace / 2.0 - np.sqrt(disc))
            roundness = float(lam_minor / lam_major) if lam_major > 1e-12 else 1.0
        else:
            roundness = 1.0

        if roundness < min_roundness:
            continue

        y_c, x_c = centers[i]
        sources.append(
            {
                "label": int(label_id),
                "x_centroid": float(x_c),
                "y_centroid": float(y_c),
                "flux": tf,
                "npix": npix,
                "roundness": roundness,
            }
        )

    # Brightest sources first — solver performs better with the bright-star population
    sources.sort(key=lambda s: s["flux"], reverse=True)
    if top_n is not None:
        sources = sources[:top_n]

    x_centroids = [s["x_centroid"] for s in sources]
    y_centroids = [s["y_centroid"] for s in sources]

    return sources, x_centroids, y_centroids
