import numpy as np
from scipy.spatial import cKDTree
from geometry import predict_pixels_from_catalog


def match_score(img_tree, pred_xy, tol_pix=25.0, sigma_pix=8.0):
    """
    Gaussian-weighted match score.

    Each predicted star contributes exp(-d² / (2σ²)) to the score, where d
    is the distance to the nearest detected source. Contributions beyond
    tol_pix are zeroed out to suppress distant noise. This is strictly better
    than the old binary count because:
    - The score surface is continuous, giving the refinement pass a real
      gradient to follow rather than flat plateaus with identical integer scores.
    - A tight cluster of nearly-perfect matches outscores a loose cluster with
      the same integer count.

    Returns
    -------
    score : float  (sum of Gaussian weights, range [0, len(pred_xy)])
    dists : array  (nearest-neighbour distances, one per catalog star)
    idx   : array  (nearest-neighbour indices into img_xy)
    """
    dists, idx = img_tree.query(pred_xy, k=1)
    weights = np.exp(-0.5 * (dists / sigma_pix) ** 2)
    weights[dists > tol_pix] = 0.0
    return float(np.sum(weights)), dists, idx


def solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix):
    """
    Solve for camera orientation (alpha, beta, gamma) by grid search + refinement.

    Improvements over the original:
    - Gaussian-weighted score instead of binary tolerance count (see match_score).
    - Coarse pass uses wide Gaussian (sigma=8 px, tol=25 px) for broad capture.
    - Before the refine pass the coarse-best solution is re-evaluated with tight
      parameters (sigma=4 px, tol=10 px) so all refinement comparisons are on
      the same scale — the refine pass cannot 'succeed' by exploiting the scale
      mismatch between passes.
    - Refine pass uses tight Gaussian (sigma=4 px, tol=10 px), which rewards
      genuine sub-pixel improvement rather than accepting anything within 25 px.
    - Post-solve diagnostics (match_fraction, rms_pix) are computed from the
      tight 10 px window so they reflect actual precision.
    - match_fraction = matched_count / n_catalog normalises the result so quality
      is comparable across images with different catalog sizes.
    """
    img_tree = cKDTree(img_xy)
    n_cat = len(cat_alt_deg)

    alpha_grid = np.deg2rad(np.arange(0, 360, 5))
    beta_grid = np.deg2rad(np.arange(0, 11, 2))
    gamma_grid = np.deg2rad(np.arange(0, 360, 20))

    best = {"score": -1.0}

    # ------------------------------------------------------------------ #
    # Coarse search — wide Gaussian for maximum capture radius            #
    # ------------------------------------------------------------------ #
    for beta in beta_grid:
        gamma_list = [0.0] if abs(beta) < 1e-12 else gamma_grid
        for gamma in gamma_list:
            for alpha in alpha_grid:
                pred_xy = predict_pixels_from_catalog(
                    cat_alt_deg, cat_az_deg, cx, cy, R_pix, alpha, beta, gamma
                )
                s, dists, idx = match_score(
                    img_tree, pred_xy, tol_pix=25.0, sigma_pix=8.0
                )
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

    # ------------------------------------------------------------------ #
    # Re-baseline the coarse winner under tight parameters.               #
    # This ensures the refine-pass comparisons are on the same scale.     #
    # ------------------------------------------------------------------ #
    s_baseline, _, _ = match_score(
        img_tree, best["pred_xy"], tol_pix=10.0, sigma_pix=4.0
    )
    best["score"] = s_baseline

    alpha_refine = np.mod(
        best["alpha"] + np.deg2rad(np.arange(-5.0, 5.0 + 1e-12, 0.5)), 2 * np.pi
    )
    beta_refine = np.clip(
        best["beta"] + np.deg2rad(np.arange(-2.0, 2.0 + 1e-12, 0.2)),
        0.0,
        np.deg2rad(15.0),
    )
    gamma_refine = np.mod(
        best["gamma"] + np.deg2rad(np.arange(-10.0, 10.0 + 1e-12, 1.0)), 2 * np.pi
    )

    # ------------------------------------------------------------------ #
    # Refine search — tight Gaussian for sharp discrimination             #
    # ------------------------------------------------------------------ #
    for beta in np.unique(beta_refine):
        gamma_list = (
            [best["gamma"]] if abs(beta) < 1e-12 else np.unique(gamma_refine)
        )
        for gamma in gamma_list:
            for alpha in np.unique(alpha_refine):
                pred_xy = predict_pixels_from_catalog(
                    cat_alt_deg, cat_az_deg, cx, cy, R_pix, alpha, beta, gamma
                )
                s, dists, idx = match_score(
                    img_tree, pred_xy, tol_pix=10.0, sigma_pix=4.0
                )
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

    # ------------------------------------------------------------------ #
    # Post-solve diagnostics                                              #
    # ------------------------------------------------------------------ #
    final_dists, _ = img_tree.query(best["pred_xy"], k=1)
    matched_mask = final_dists <= 10.0
    matched_count = int(np.sum(matched_mask))

    best["matched_count"] = matched_count
    best["match_fraction"] = matched_count / n_cat if n_cat > 0 else 0.0
    best["rms_pix"] = (
        float(np.sqrt(np.mean(final_dists[matched_mask] ** 2)))
        if matched_count > 0
        else np.nan
    )

    return best
