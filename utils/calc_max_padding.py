import gunpowder as gp
import numpy as np


def calc_max_padding(
        output_size,
        voxel_size,
        neighborhood=None,
        sigma=None,
        mode='shrink'):

    '''Calculate maximum labels padding needed.

    Args:
		output_size (array-like of ``int``):
            output size of network, in world units (a gunpowder coordinate)
        voxel_size (array-like of ``int``):
            voxel size to use (a gunpowder coordinate)
        neighborhood (``list`` of array-like, optional):
            affinity neighborhood to use.
        sigma (``int``, optional):
            sigma if using lsds
        mode (``string``, optional):
            mode to use for snapping roi to grid, see gunpowder roi
            documentation for details

    Explanation:
        when padding labels, we need to ensure that each batch still contains at
        least 50% of GT data. Additionally, we need to also consider worst case
        45 degree rotation when elastically augmenting the data. Our max padding
        is calculated as follows:
            output_size = output size of network in world coordinates (i.e \
                    nanometers not voxels)
            method_padding = largest affinity neighborhood * voxel size (for \
                    affinities) or sigma * voxel size (for lsds)
            diagonal = diagonal between x and y dimensions (i.e square root \ of
            the sum of squares of x and y axes)
            max_padding = (output_size[z]/2, diagonal/2, diagonal/2) + \
                    method_padding
        we then need to ensure max padding is a multiple of the voxel size - use
        snap_to_grid for this (see gunpowder.roi.snap_to_grid())
    '''

    if neighborhood is not None:

        if len(neighborhood) > 3:
            neighborhood = neighborhood[9:12]

        max_affinity = gp.Coordinate(
                            [np.abs(aff) for val in neighborhood \
                                    for aff in val if aff != 0]
                        )

        method_padding = voxel_size * max_affinity

    if sigma:

        method_padding = gp.Coordinate((sigma*3,)*3)

    diag = np.sqrt(output_size[1]**2 + output_size[2]**2)

    max_padding = gp.Roi(
                    (gp.Coordinate(
                        [i/2 for i in [output_size[0], diag, diag]]) +
                        method_padding),
                    (0,)*3).snap_to_grid(voxel_size,mode=mode)

    return max_padding.get_begin()
