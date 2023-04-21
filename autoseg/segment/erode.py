import numpy as np
from scipy.ndimage import binary_erosion

def erode(labels, steps, only_xy=True):
    
    if only_xy:
        assert len(labels.shape) == 3
        for z in range(labels.shape[0]):
            labels[z] = erode(labels[z], steps, only_xy=False)
        return labels

    # get all foreground voxels by erosion of each component
    foreground = np.zeros(shape=labels.shape, dtype=bool)
    
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = labels==label
        # Assume that masked out values are the same as the label we are
        # eroding in this iteration. This ensures that at the boundary to
        # a masked region the value blob is not shrinking.
        eroded_label_mask = binary_erosion(label_mask, iterations=steps, border_value=1)

        foreground = np.logical_or(eroded_label_mask, foreground)

    # label new background
    background = np.logical_not(foreground)
    labels[background] = 0

    return labels
