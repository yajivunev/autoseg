import os
import sys
import gunpowder as gp
import numpy as np


def train(
        iterations,
        sources,
        model_path,
        checkpoint_path,
        log_dir,
        snapshots_dir,
        increase=None):


    # import model from model_path and make instance
    sys.path.append(os.path.dirname(model_path))
    from model import model, loss, optimizer, input_shape, output_shape
    from model import voxel_size as default_voxel_size
    from model import training_array_keys as keys

    # from model import pipeline ???
    # from model import affs neighborhood ??? lsds sigma ???

    model.train()

    # make ArrayKeys

    # I/O shapes and sizes
    if increase is not None:
        increase = gp.Coordinate(increase)
    else:
        increase = gp.Coordinate([0,]*len(input_shape))

    input_shape = gp.Coordinate(input_shape) + increase
    output_shape = gp.Coordinate(output_shape) + increase

    voxel_size = open_ds(sources[0][0],sources[0][1]).voxel_size
    default_voxel_size = gp.Coordinate(default_voxel_size)

    # XY downsample factor
    if downsample==True:
        downsample = int(round(default_voxel_size[-1]/voxel_size[-1]))
    elif type(downsample) == int:
        pass
    else:
        downsample = 1

    downsample_factors = (1,) * (len(input_shape) - 2) + (downsample,downsample)
    voxel_size = voxel_size * gp.Coordinate(downsample_factors) if downsample > 1 else voxel_size
    
    # world units (nm)
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    context = (input_size - output_size) // 2

    # Add to gp Request

    # sources

    # augmentations

    # learning targets

    # scale-shits, unsqueezes, stacks, PreCache

    # Train node

    # scale-shift, Snapshots
