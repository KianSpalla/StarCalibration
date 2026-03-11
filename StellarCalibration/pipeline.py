import numpy as np
from GONet_Wizard.GONet_utils import GONetFile
from detection import find_stars, find_centroids
from query import query_catalog_altaz_from_meta
from geometry import filter_image_sources_by_radius
from solver import solve_orientation
from centering import find_zenith_pixel_and_center, build_shifted_image


def run_calibration(imagePath, show_plots=False, N=5, gmax=2.5):
    go = GONetFile.from_file(imagePath)
    go.remove_overscan()
    img = go.green

    imgMean = float(np.mean(img))
    imgStd = float(np.std(img))
    mask = img > imgMean + N * imgStd

    labels, numLabels = find_stars(np.array(mask, dtype=bool))
    xCentroids, yCentroids = find_centroids(img, labels, numLabels)
    imgXY = np.column_stack([xCentroids, yCentroids])

    cx, cy = 1030, 760
    radiusPix = 740
    catalogRadiusDeg = 60.0

    imgXY = filter_image_sources_by_radius(
        imgXY=imgXY, cx=cx, cy=cy, radiusPix=radiusPix, radiusDeg = catalogRadiusDeg,
    )

    catalogAltDeg, catalogAzDeg, _ = query_catalog_altaz_from_meta(
        go.meta, radiusDeg=catalogRadiusDeg, gmax=gmax, top_m=None,
    )

    best = solve_orientation(imgXY, catalogAltDeg, catalogAzDeg, cx, cy, radiusPix)

    centerResult = find_zenith_pixel_and_center(
        img=img, best=best, cx=cx, cy=cy, radiusPix=radiusPix,
    )

    print(f"catalog_stars={len(catalogAltDeg)}, image_sources={len(imgXY)}")
    print(f"score={best['score']}, matched={best['matched_count']}, rms_pix={best['rms_pix']:.3f}")
    print(
        "alpha_deg={:.3f}, beta_deg={:.3f}, gamma_deg={:.3f}".format(
            np.rad2deg(best["alpha"]),
            np.rad2deg(best["beta"]),
            np.rad2deg(best["gamma"]),
        )
    )
    print(f"Zenith pixel: x={centerResult['zenithX']:.2f}, y={centerResult['zenithY']:.2f}")
    print(f"Applied shift: dx={centerResult['shiftX']:.2f}, dy={centerResult['shiftY']:.2f}")

    if show_plots:
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots()
        ax1.imshow(img, origin="upper", cmap="gray",
                   vmin=imgMean - 2 * imgStd, vmax=imgMean + 5 * imgStd)
        ax1.scatter(imgXY[:, 0], imgXY[:, 1], s=50, edgecolor="red",
                    facecolor="none", label="Detected sources")
        ax1.scatter(best["predictedXY"][:, 0], best["predictedXY"][:, 1], s=50,
                    edgecolor="blue", facecolor="none", label="Catalog predictions")
        ax1.scatter([centerResult["targetCenterX"]], [centerResult["targetCenterY"]],
                    s=100, marker="+", c="yellow", label="Image centre")
        ax1.scatter([centerResult["zenithX"]], [centerResult["zenithY"]],
                    s=120, marker="x", c="cyan", label="Zenith")
        ax1.plot(
            [centerResult["zenithX"], centerResult["targetCenterX"]],
            [centerResult["zenithY"], centerResult["targetCenterY"]],
            color="cyan", linestyle="--", linewidth=1.5, label="Applied shift",
        )
        ax1.legend()
        ax1.set_title(f"Orientation solve \u2014 score: {best['score']} matches")
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.imshow(centerResult["centered_sub"], origin="upper", cmap="gray",
                   vmin=imgMean - 2 * imgStd, vmax=imgMean + 5 * imgStd)
        ax2.scatter([centerResult["targetCenterX"]], [centerResult["targetCenterY"]],
                    s=120, marker="x", c="cyan", label="Zenith (centred)")
        ax2.legend()
        ax2.set_title("Shifted image \u2014 zenith at centre")
        plt.show()

    shiftedResult = build_shifted_image(
        imagePath=imagePath,
        shiftX=centerResult["shiftX"],
        shiftY=centerResult["shiftY"],
    )
    print("Shifted image prepared (not yet saved).")

    return {
        "best": best,
        "centerResult": centerResult,
        "img": img,
        "imgXY": imgXY,
        "shiftedImage": shiftedResult,
        "shifted_image": shiftedResult,
        "shiftedFormat": "PNG",
        "shifted_format": "PNG",
        "suggested_suffix": ".png",
    }

if __name__ == "__main__":
    run_calibration(r"C:\Users\spall\Desktop\GONet\StarCalibration\Testing Images\256_251029_204008_1761770474.jpg", show_plots=True, N=5, gmax=2.5)