import sys
import numpy as np
import zarr
from funlib.persistence import open_ds,prepare_ds
from funlib.geometry import Coordinate, Roi

""" Script to make a ROI cutout of a zarr container's datasets. """

pad = None # optional. set to None to disable

if __name__ == "__main__":

    f = sys.argv[1] # input zarr
    out_f = sys.argv[2] # output zarr
    
    offset = tuple([int(o) for o in sys.argv[3:6]]) # voxel units z y x
    shape = tuple([int(s) for s in sys.argv[6:9]]) # voxel units z y x

    datasets = sys.argv[9:] # list of datasets
    
    for ds in datasets:

        old_ds = open_ds(f,ds)
        resolution = old_ds.voxel_size 
        dtype = old_ds.dtype

        roi = Roi(
                [o*r for o,r in zip(offset,resolution)],
                [s*r for s,r in zip(shape,resolution)]
                )
        
        if pad:
            roi = roi.grow(pad,pad)
            
        roi = roi.snap_to_grid(resolution)  
 
        print(f"{ds}")
        print(f"{offset}, {shape}")
        print(f"{roi}")
        print(f"{resolution}")
        print(f"{dtype}")
        print(" ")
        
        out_ds = prepare_ds(
                out_f,
                ds,
                roi,
                resolution,
                dtype,
                compressor={'id': 'blosc', 'clevel': 3})

        out_ds[roi] = open_ds(f,ds).to_ndarray(roi, fill_value=0)
