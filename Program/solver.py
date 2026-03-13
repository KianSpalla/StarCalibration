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

NOTE: A more in depth explanation of the alpha, beta, and gamma angles can be found in comments at the bottom of this file.
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


"""
To understand the Alpha, Beta, and Gamma values, it is important to fully understand the problem we are facing. 
This problem is the incorrect placement of GONet Cameras. There are two conditions for a GONet to be considered "Calibrated"
    1. The GONet camera must be facing directly north
    2. The GONet camera must be perfectly level

Alpha, Beta, and Gamma tie directly to these two issues.
    Rule 1 is solved through Alpha
    Rule 2 is solved by Beta and Gamma in unison

As there are two rules for a GONet being calibrated, every solution involves two steps
    1. Rotating the camera so that North is centered at the top (Alpha)
    2. Tilting the camera so that the zenith is at the center of the image (Beta + Gamma)

Alpha is independent of Beta + Gamma
-Alpha
    Alpha is the rotation about the optical axis. In the sense of correcting GONet orientation, 
    this would be pointing the arrows on a GONet so that they are directly north

Beta + Gamma are tied together
-Beta
    Beta is the tilt away from level. In the sense of correcting GONet orientation, 
    this would be how much the camera is tipped.
-Gamma
    Gamma is the direction of the tilt. In the sense of correcting GONet orientation, 
    this would be which cardinal direction (North, East, West, South, or anything in between) the tilt points towards

This is why when beta is 0, gamma is irrelevant, as if there is no tilt away from level,
it doesn't matter which direction the nonexistent tilt points towards

When we are solving for this problem in the solve_orientation function, we are searching over these Alpha, Beta, and Gamma values
and looking for the combination of these three values that gives us the most amount of matches.
"""