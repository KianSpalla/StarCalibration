import numpy as np
from scipy.ndimage import (
    label as nd_label,
    sum as ndimage_sum,
    center_of_mass as ndimage_com,
)


def compact_search(mask):
    labels, num_clusters = nd_label(mask)
    return labels, num_clusters

def measure_sources(img, labels, num_clusters):
    if num_clusters == 0:
        return [], [], []

    label_ids = np.arange(1, num_clusters + 1)
    total_fluxes = ndimage_sum(img, labels, index=label_ids)
    centers = ndimage_com(img, labels, index=label_ids)

    sources = []
    x_centroids = []
    y_centroids = []

    for i, label_id in enumerate(label_ids):
        tf = float(total_fluxes[i])
        if tf <= 0:
            continue

        y_c, x_c = centers[i]

        sources.append(
            {
                "label": int(label_id),
                "x_centroid": float(x_c),
                "y_centroid": float(y_c),
                "flux": tf,
            }
        )
        x_centroids.append(float(x_c))
        y_centroids.append(float(y_c))

    return sources, x_centroids, y_centroids