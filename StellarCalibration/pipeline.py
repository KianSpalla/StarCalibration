import numpy as np
from GONet_Wizard.GONet_utils import GONetFile
from stardetection import compact_search, measure_sources
from starquery import query_catalog_altaz_from_meta
from geometry import filter_image_sources_by_radius
from solver import solve_orientation
from centering import find_zenith_pixel_and_center, build_shifted_image_same_format


def run_calibration(image_path, show_plots=False, N=5, gmax=2.5):
    go = GONetFile.from_file(image_path)
    go.remove_overscan()
    sub = go.green

    sub_mean = float(np.mean(sub))
    sub_std = float(np.std(sub))
    mask = sub > sub_mean + N * sub_std

    labels, num_labels = compact_search(np.array(mask, dtype=bool))
    _, x_centroids, y_centroids = measure_sources(sub, labels, num_labels)
    img_xy = np.column_stack([x_centroids, y_centroids])

    cx, cy = 1030, 760
    R_pix = 740
    catalog_radius_deg = 60.0

    img_xy = filter_image_sources_by_radius(
        img_xy=img_xy, cx=cx, cy=cy, R_pix=R_pix, radius_deg=catalog_radius_deg,
    )
    if len(img_xy) == 0:
        raise RuntimeError(
            "No image centroids remain after sky-radius filtering. "
            "Check that cx/cy/R_pix match the actual image geometry."
        )

    cat_alt_deg, cat_az_deg, _ = query_catalog_altaz_from_meta(
        go.meta, radius_deg=catalog_radius_deg, gmax=gmax, top_m=None,
    )
    if len(cat_alt_deg) == 0:
        raise RuntimeError(
            "No catalog stars found above the horizon. "
            "Check that the GPS coordinates and observation time in the "
            "image metadata are correct."
        )

    best = solve_orientation(img_xy, cat_alt_deg, cat_az_deg, cx, cy, R_pix)

    center_result = find_zenith_pixel_and_center(
        sub=sub, best=best, cx=cx, cy=cy, R_pix=R_pix,
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
    print(f"Zenith pixel: x={center_result['zenith_x']:.2f}, y={center_result['zenith_y']:.2f}")
    print(f"Applied shift: dx={center_result['shift_x']:.2f}, dy={center_result['shift_y']:.2f}")

    if show_plots:
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots()
        ax1.imshow(sub, origin="lower", cmap="gray",
                   vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
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
        ax1.set_title(f"Orientation solve \u2014 score: {best['score']} matches")
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.imshow(center_result["centered_sub"], origin="lower", cmap="gray",
                   vmin=sub_mean - 2 * sub_std, vmax=sub_mean + 5 * sub_std)
        ax2.scatter([center_result["target_cx"]], [center_result["target_cy"]],
                    s=120, marker="x", c="cyan", label="Zenith (centred)")
        ax2.legend()
        ax2.set_title("Shifted image \u2014 zenith at centre")
        plt.show()

    shifted_result = build_shifted_image_same_format(
        image_path=image_path,
        shift_x=center_result["shift_x"],
        shift_y=center_result["shift_y"],
    )
    print("Shifted image prepared (not yet saved).")

    return {
        "best": best,
        "center_result": center_result,
        "shifted_image": shifted_result["shifted_image"],
        "shifted_format": shifted_result["shifted_format"],
        "suggested_suffix": shifted_result["suggested_suffix"],
    }
