import sys
import numpy as np
import zarr


def edge_slices(arr):

    shape = arr.shape
    ndim = arr.ndim
    slices = []

    for i in range(ndim):
        sl = [slice(None)] * ndim
        sl[i] = slice(0, 1)
        slices.append(tuple(sl))
        sl = [slice(None)] * ndim
        sl[i] = slice(shape[i]-1, shape[i])
        slices.append(tuple(sl)) 
    return slices


def get_edge_labels(arr):
    
    slices = edge_slices(arr)
    edge_faces = [arr[sl] for sl in slices]
    edge_labels = np.concatenate([np.unique(label_arr) for label_arr in edge_faces])
    
    return np.unique(edge_labels)


def filter_labels(zarr_container, labels_dataset):

    f = zarr.open(zarr_container, "a")

    seg = f[labels_dataset][:]
    shape = seg.shape
    
    unique = np.unique(seg, return_counts=True)

    # veto IDs that are touching the edge
    veto = set(get_edge_labels(seg))

    remove = []

    for i, j in zip(unique[0], unique[1]):
        if j < 150000 and j not in veto:
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
