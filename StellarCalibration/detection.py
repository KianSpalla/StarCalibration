import numpy as np
from scipy.ndimage import (
    label as nd_label,
    sum as nd_sum,
    center_of_mass as nd_com,
)


def find_stars(mask):
    labels, numClusters = nd_label(mask)
    return labels, numClusters

def find_centroids(img, labels, numClusters):
    if numClusters == 0:
        return [], [], []

    labelIDs = np.arange(1, numClusters + 1)
    totalFluxes = nd_sum(img, labels, index=labelIDs)
    centers = nd_com(img, labels, index=labelIDs)

    xCentroids = []
    yCentroids = []

    for i, label_id in enumerate(labelIDs):
        tf = float(totalFluxes[i])
        if tf <= 0:
            continue

        yCenters, xCenters = centers[i]

        xCentroids.append(float(xCenters))
        yCentroids.append(float(yCenters))

    return xCentroids, yCentroids