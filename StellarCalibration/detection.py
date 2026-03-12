import numpy as np
from scipy.ndimage import (
    label as nd_label,
    sum as nd_sum,
    center_of_mass as nd_com,
)

"""
dynamic_find_stars(img)
PROBELM: when we createa mask using a threshold based on the entire image,
    some stars in brighter areas of the sky like the center of the image or nearcity horizons/clouds
    get drowned out and are not captured by the mask.

SOLUTION: instead of creating a threshold based on the entire image, we can split the image into sections,
    then we take the thresholds of those individual sections to build the mask. This eliminates the variance in overall brightness of different sections, 
    and instead we are choosing stars based on contrast to their respective background.

PSEUDO:
    dynamic_find_stars(img):
        create a mask variable we will add to

        Split image into sections
            per section:
            create mask for section
            call cluster_stars
            add results into labels and numClusters
    
    return labels and numClusters

img is a 2d array, we can split image into 120x120 pixel grids

"""
def dynamic_find_stars(img, N = 5, sectionSize = 200):
    labels = np.zeros(img.shape[:2], dtype=int)
    numClusters = 0

    for r in range(0, img.shape[0], sectionSize):
        for c in range(0, img.shape[1], sectionSize):
            section = img[r:r+sectionSize, c:c+sectionSize]
            sectionMean = np.mean(section)
            sectionStd = np.std(section)
            mask = section > sectionMean + N * sectionStd
            sectionLabels, sectionNumClusters = cluster_stars(mask)
            if sectionNumClusters > 0:
                labels[r:r+section.shape[0], c:c+section.shape[1]] = np.where(
                    sectionLabels > 0,
                    sectionLabels + numClusters,
                    0,
                )
            numClusters += sectionNumClusters

    return labels, numClusters


"""
find_stars takes a mask as input, uses nd_label to create stars labeled from 1 to N.
returns labels which is a array of the connected values in the mask.
returns numClusters, which is the number of clustered components created during the function.
"""

def cluster_stars(mask):
    labels, numClusters = nd_label(mask)
    return labels, numClusters

"""
find_centroids takes the image, and the output from find_stars (labels and numClusters) 
and creates weighted centroids on each star cluster. Returns xCentroids which is an array that holds the x values of each cluster,
and yCentroids that holds the y values of each clusters. The indicies of the two arrays coorelate with one another.
"""
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