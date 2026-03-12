import numpy as np
from scipy.spatial import cKDTree
from geometry import predict_pixels_from_catalog

"""
match_score takes a cKDTree of the image sources, the predicted pixel positions from the catalog, and a pixel tolerance.
It queries the tree to find the nearest image source for each predicted position and counts how many are within the pixel tolerance.
Returns the total score (number of matches), the distances to the nearest image sources, and their indices.
"""
def match_score(imgTree, predictedXY, pixelTolerance=20.0):
    starDistance, starIndex = imgTree.query(predictedXY, k=1)
    score = np.sum(starDistance <= pixelTolerance)
    return int(score), starDistance, starIndex

"""
solve_orientation takes the detected image sources, the catalog altitudes and azimuths, the center coordinates, and the radius in pixels.
It performs a search using alpha, beta, and gamma angles to find the best orientation that matches the detected sources to the catalog predictions.
It first does a coarse search over a grid of angles, then refines the search around the best solution found. 
It returns the best orientation parameters, the score, and the predicted pixel positions for the catalog stars.
"""
def solve_orientation(imgXY, catalogAltDeg, catalogAzDeg, cx, cy, radiusPix):
    imgTree = cKDTree(imgXY)

    alphaGrid = np.deg2rad(np.arange(0, 360, 5))
    betaGrid = np.deg2rad(np.arange(0, 11, 2))
    gammaGrid = np.deg2rad(np.arange(0, 360, 20))

    best = {"score": -1}

    for beta in betaGrid:
        gammaList = [0.0] if abs(beta) < 1e-12 else gammaGrid
        for gamma in gammaList:
            for alpha in alphaGrid:
                predictedXY = predict_pixels_from_catalog(catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, alpha, beta, gamma)
                score, starDistance, starIndex = match_score(imgTree, predictedXY, pixelTolerance=25.0)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "starDistance": starDistance,
                        "starIndex": starIndex,
                        "predictedXY": predictedXY,
                    }

    alphaRefine = best["alpha"] + np.deg2rad(np.arange(-5.0, 5.0 + 1e-12, 0.5))
    betaRefine = best["beta"] + np.deg2rad(np.arange(-2.0, 2.0 + 1e-12, 0.2))
    gammaRefine = best["gamma"] + np.deg2rad(np.arange(-10.0, 10.0 + 1e-12, 1.0))

    alphaRefine = np.mod(alphaRefine, 2 * np.pi)
    gammaRefine = np.mod(gammaRefine, 2 * np.pi)
    betaRefine = np.clip(betaRefine, 0.0, np.deg2rad(15.0))

    for beta in np.unique(betaRefine):
        gammaList = [best["gamma"]] if abs(beta) < 1e-12 else np.unique(gammaRefine)
        for gamma in gammaList:
            for alpha in np.unique(alphaRefine):
                predictedXY = predict_pixels_from_catalog(catalogAltDeg, catalogAzDeg, cx, cy, radiusPix, alpha, beta, gamma)
                score, starDistance, starIndex = match_score(imgTree, predictedXY, pixelTolerance=25.0)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "starDistance": starDistance,
                        "starIndex": starIndex,
                        "predictedXY": predictedXY,
                    }

    matchedMask = best["starDistance"] <= 25.0
    matchedCount = int(np.sum(matchedMask))
    if matchedCount > 0:
        best["rms_pix"] = float(np.sqrt(np.mean(best["starDistance"][matchedMask] ** 2)))
    else:
        best["rms_pix"] = np.nan
    best["matched_count"] = matchedCount
    
    return best