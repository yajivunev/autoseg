import sys
import numpy as np
import zarr


def remove_labels(zarr_container, labels_dataset, remove):

    f = zarr.open(zarr_container, "a")

    seg = f[labels_dataset][:]
    shape = seg.shape
    
    print(f"Removing {remove}..")

    seg[np.isin(seg, remove)] = 0

    f[f'{labels_dataset}_filtered'] = seg
    f[f'{labels_dataset}_filtered'].attrs['offset'] = f[labels_dataset].attrs['offset']
    f[f'{labels_dataset}_filtered'].attrs['resolution'] = f[labels_dataset].attrs['resolution']


if __name__ == "__main__":

    f = sys.argv[1]
    ds = sys.argv[2]
    remove = list(map(int,sys.argv[3:]))

    remove_labels(f,ds,remove)
