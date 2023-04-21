import sys
import numpy as np
import zarr


def make_mask(f,ds,out_ds=None):

    f = zarr.open(f,"r+")

    arr = f[ds][:]
    offset = f[ds].attrs["offset"]
    res = f[ds].attrs["resolution"]

    unlabelled = (arr > 0).astype(np.uint8)

    # write
    if out_ds is None:
        out_ds = "unlabelled"

    f[out_ds] = unlabelled
    f[out_ds].attrs["offset"] = offset
    f[out_ds].attrs["resolution"] = res


if __name__ =="__main__":

    f = sys.argv[1]
    ds = sys.argv[2]

    make_mask(f,ds)
