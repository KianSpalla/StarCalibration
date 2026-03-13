import numpy as np
from GONet_Wizard.GONet_utils import GONetFile
from detection_improved import adaptive_threshold_mask, compact_search, measure_sources
from query import query_catalog_altaz_from_meta
from geometry import filter_image_sources_by_radius
from solver_improved import solve_orientation
from centering import find_zenith_pixel_and_center, build_shifted_image


def run_calibration(image_path, show_plots=False, n_sigma=5.0, gmax=2.5, top_n=200):
    go = GONetFile.from_file(image_path)
    go.remove_overscan()
    img = go.green

    # Adaptive threshold using per-tile median + NMAD instead of a single global threshold.
    # This handles vignetting and background gradients across the image.
    mask, background, noise = adaptive_threshold_mask(img, tile_size=128, n_sigma=n_sigma)

    labels, num_labels = compact_search(np.array(mask, dtype=bool))

    # Background-subtracted centroids, shape/size filtering, roundness filtering,
    # and flux-ranked capping — only the best top_n sources go to the solver.
    sources, x_centroids, y_centroids = measure_sources(
        img,
        labels,
        num_labels,
        background=background,
        min_pixels=3,
        max_pixels=200,
        min_roundness=0.3,
        top_n=top_n,
    )
    img_xy = np.column_stack([x_centroids, y_centroids]) if x_centroids else np.empty((0, 2))

    cx, cy = 1030, 760
    R_pix = 740
    catalog_radius_deg = 60.0

    img_xy = filter_image_sources_by_radius(
        img_xy=img_xy, cx=cx, cy=cy, R_pix=R_pix, radius_deg=catalog_radius_deg,
    )

    cat_alt_deg, cat_az_deg, _ = query_catalog_altaz_from_meta(
        go.meta, radius_deg=catalog_radius_deg, gmax=gmax, top_m=None,
    )

    best = solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix)

    center_result = find_zenith_pixel_and_center(
        img=img, best=best, cx=cx, cy=cy, R_pix=R_pix,
    )

    img_mean = float(np.mean(img))
    img_std = float(np.std(img))

    print(f"catalog_stars={len(cat_alt_deg)}, image_sources={len(img_xy)}")
    print(
        f"score={best['score']:.3f}, matched={best['matched_count']}, "
        f"match_fraction={best['match_fraction']:.3f}, rms_pix={best['rms_pix']:.3f}"
    )
    print(
        "alpha_deg={:.3f}, beta_deg={:.3f}, gamma_deg={:.3f}".format(
            np.rad2deg(best["alpha"]),
            np.rad2deg(best["beta"]),
            np.rad2deg(best["gamma"]),
        )
    )
    print(f"Zenith pixel: x={center_result['zenith_x']:.2f}, y={center_result['zenith_y']:.2f}")
    print(f"Applied shift: dx={center_result['shift_x']:.2f}, dy={center_result['shift_y']:.2f}")

    if show_plots:
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots()
        ax1.imshow(img, origin="upper", cmap="gray",
                   vmin=img_mean - 2 * img_std, vmax=img_mean + 5 * img_std)
        ax1.scatter(img_xy[:, 0], img_xy[:, 1], s=50, edgecolor="red",
                    facecolor="none", label="Detected sources")
        ax1.scatter(best["pred_xy"][:, 0], best["pred_xy"][:, 1], s=50,
                    edgecolor="blue", facecolor="none", label="Catalog predictions")
        ax1.scatter([center_result["target_cx"]], [center_result["target_cy"]],
                    s=100, marker="+", c="yellow", label="Image centre")
        ax1.scatter([center_result["zenith_x"]], [center_result["zenith_y"]],
                    s=120, marker="x", c="cyan", label="Zenith")
        ax1.plot(
            [center_result["zenith_x"], center_result["target_cx"]],
            [center_result["zenith_y"], center_result["target_cy"]],
            color="cyan", linestyle="--", linewidth=1.5, label="Applied shift",
        )
        ax1.legend()
        ax1.set_title(
            f"Orientation solve \u2014 score: {best['score']:.2f}, "
            f"matched: {best['matched_count']}/{len(cat_alt_deg)} "
            f"({best['match_fraction']:.1%})"
        )
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.imshow(center_result["centered_sub"], origin="upper", cmap="gray",
                   vmin=img_mean - 2 * img_std, vmax=img_mean + 5 * img_std)
        ax2.scatter([center_result["target_cx"]], [center_result["target_cy"]],
                    s=120, marker="x", c="cyan", label="Zenith (centred)")
        ax2.legend()
        ax2.set_title("Shifted image \u2014 zenith at centre")
        plt.show()

    shifted_result = build_shifted_image(
        image_path=image_path,
        shift_x=center_result["shift_x"],
        shift_y=center_result["shift_y"],
    )
    print("Shifted image prepared (not yet saved).")

    return {
        "best": best,
        "center_result": center_result,
        "img": img,
        "img_xy": img_xy,
        "sources": sources,
        "shifted_image": shifted_result,
    }


if __name__ == "__main__":
    run_calibration(
        r"c:\Users\spall\Documents\GitHub\StarCalibration\Testing Images\256_251029_204008_1761770474.jpg",
        show_plots=True,
        n_sigma=5.0,
        gmax=2.5,
        top_n=200,
    )
