import numpy as np
from scipy.ndimage import distance_transform_edt


def expand_labels(labels):

    distance = labels.shape[0]

    distances, indices = distance_transform_edt(
            labels == 0,
            return_indices=True)

    expanded_labels = np.zeros_like(labels)

    dilate_mask = distances <= distance

    masked_indices = [
            dimension_indices[dilate_mask]
            for dimension_indices in indices
    ]

    nearest_labels = labels[tuple(masked_indices)]

    expanded_labels[dilate_mask] = nearest_labels

    return expanded_labels
