import numpy as np
from scipy.ndimage import (
    label,
    sum,
    center_of_mass,
    binary_opening
)

"""
dynamic_find_stars(img)
PROBELM: when we create a mask using a threshold based on the entire image,
    some stars in brighter areas of the sky, such as the center of the image or nearcity horizons/clouds,
    get drowned out and are not captured by the mask.

SOLUTION: instead of creating a threshold based on the entire image, we can split the image into sections,
    then we take the thresholds of those individual sections to build the mask. 
    This eliminates the variance in overall brightness of different sections, 
    and instead we are choosing stars based on contrast to their respective backgrounds.

NOTE: we dont want to allow too many stars to pass the threshold until we have a better matching methodology as we can easily make false matches

PSEUDO:
    dynamic_find_stars(img):
        Split image into sections
            per section:
            create mask for section
            call cluster_stars
            add results into labels and numClusters
    
    return labels and numClusters

TODO:
    Somehow implement a way to filter stars by a min and max size. 
    This will help get rid of outliers in noise and objects that are too big to be stars like clouds
"""
def dynamic_find_stars(img, N = 5, sectionSize = 200):
    labels = np.zeros(img.shape[:2], dtype=int)
    numClusters = 0

    for r in range(0, img.shape[0], sectionSize):
        for c in range(0, img.shape[1], sectionSize):
            section = img[r:r+sectionSize, c:c+sectionSize]
            sectionMed = np.median(section)
            sectionNMAD = 1.4826 * np.median(np.abs(section - sectionMed))
            mask = section > sectionMed + N * sectionNMAD
            #mask = binary_opening(mask) 
            """
            This function (binary_opening) is still unclear to me, it seems to work fairly well for removing noise outliers
            The premise of the function is that it takes a mask and removes outliers like isolated true values
            keeping only the large compact objects, With this said, I am unsure on the threshold it uses to do this
            and whether it is the best solution for this project or not.

            We can probably replace this function with min and max filtering once i figure that out, 
            it should essentially do the same exact thing while also being more adjustable but for now this seems to work well enough.
            """
            sectionLabels, sectionNumClusters = cluster_stars(mask)
            labels[r:r+section.shape[0], c:c+section.shape[1]] = np.where(sectionLabels > 0,sectionLabels + numClusters,0)
            numClusters += sectionNumClusters

    return labels, numClusters


"""
cluster_stars takes a mask as input, uses nd_label to create stars labeled from 1 to N.
returns labels which is a array of the connected values in the mask.
returns numClusters, which is the number of clustered components created during the function.
"""

def cluster_stars(mask):
    labels, numClusters = label(mask)
    return  labels, numClusters

"""
filter_by_size removes blobs from labels that are smaller than minPixels or larger than maxPixels.
minPixels filters out hot pixels and noise; maxPixels filters out clouds and other large non-stellar objects.
Returns updated labels and numClusters with the surviving blobs relabeled from 1 to N.
"""
def filter_by_size(labels, numClusters, minPixels=4, maxPixels=200):
    if numClusters == 0:
        return labels, numClusters

    labelIDs = np.arange(1, numClusters + 1)
    ones = np.ones_like(labels, dtype=float)
    pixelCounts = sum(ones, labels, index=labelIDs)

    filtered = np.zeros_like(labels)
    newID = 0
    for i, lid in enumerate(labelIDs):
        if minPixels <= int(pixelCounts[i]) <= maxPixels:
            newID += 1
            filtered[labels == lid] = newID

    return filtered, newID

"""
find_centroids takes the image, and the output from find_stars (labels and numClusters) 
and creates weighted centroids on each star cluster. Returns xCentroids which is an array that holds the x values of each cluster,
and yCentroids that holds the y values of each clusters. The indicies of the two arrays coorelate with one another.
"""
def find_centroids(img, labels, numClusters):
    if numClusters == 0:
        return [], []

    background = np.median(img)

    img_sub = img.astype(float) - background

    labelIDs = np.arange(1, numClusters + 1)
    totalFluxes = sum(img_sub, labels, index=labelIDs)
    centers = center_of_mass(img_sub, labels, index=labelIDs)

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