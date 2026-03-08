import numpy as np
from scipy.spatial import cKDTree
from geometry import predict_pixels_from_catalog


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