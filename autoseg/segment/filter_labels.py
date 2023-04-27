import sys
import numpy as np
import zarr

def filter_labels(zarr_container, labels_dataset):

    f = zarr.open(zarr_container, "a")

    seg = f[labels_dataset][:]
    
    unique = np.unique(seg, return_counts=True)

    remove = []

    for i, j in zip(unique[0], unique[1]):
        if j < 25000:
            remove.append(i)

    print(f"Removing {len(remove)} fragments out of {len(unique[0])}..")

    seg[np.isin(seg, remove)] = 0

    f[f'{labels_dataset}_filtered'] = seg
    f[f'{labels_dataset}_filtered'].attrs['offset'] = f[labels_dataset].attrs['offset']
    f[f'{labels_dataset}_filtered'].attrs['resolution'] = f[labels_dataset].attrs['resolution']

if __name__ == "__main__":

	f = sys.argv[1]
	ds = sys.argv[2]

	filter_labels(f,ds)
