import numpy as np
import daisy

from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation import MWSGridGraph, compute_mws_clustering
from typing import Optional

from skimage.transform import rescale

from watershed import watershed_from_boundary_distance

def mutex_watershed(
        affs,
        offsets,
        stride,
        algorithm="kruskal",
        mask=None,
        randomize_strides=True,
        sep=3) -> np.ndarray:

    affs = 1 - affs

    affs[:sep] = affs[:sep] * -1
    affs[:sep] = affs[:sep] + 1

    segmentation = compute_mws_segmentation(
        affs,
        offsets,
        sep,
        strides=stride,
        randomize_strides=randomize_strides,
        algorithm=algorithm,
        mask=mask,
    )

    return segmentation


def seeded_mutex_watershed(
        seeds,
        affs,
        offsets,
        mask,
        stride,
        randomize_strides=True) -> np.ndarray:
    
    shape = affs.shape[1:]
    if seeds is not None:
        assert (len(seeds.shape) == len(shape)
        ), f"Got shape {seeds.data.shape} for mask but expected {shape}"
    if mask is not None:
        assert (len(mask.shape) == len(shape)
        ), f"Got shape {mask.data.shape} for mask but expected {shape}"

    grid_graph = MWSGridGraph(shape)
    if seeds is not None:
        grid_graph.update_seeds(seeds.data)

    ndim = len(offsets[0])

    grid_graph.add_attractive_seed_edges = True
    neighbor_affs, lr_affs = (
        np.require(affs[:ndim], requirements="C"),
        np.require(affs[ndim:], requirements="C"),
    )
    
    # assuming affinities are 1 between voxels that belong together and
    # 0 if they are not part of the same object. Invert if the other way
    # around.
    # neighbors_affs should be high for objects that belong together
    # lr_affs is the oposite
    lr_affs = 1 - lr_affs

    uvs, weights = grid_graph.compute_nh_and_weights(
        neighbor_affs, offsets[:ndim]
    )

    if stride is None:
        stride = [1] * ndim
        
    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
        lr_affs,
        offsets[ndim:],
        stride,
        randomize_strides=randomize_strides,
    )

    # compute the segmentation
    n_nodes = grid_graph.n_nodes
    segmentation = compute_mws_clustering(
        n_nodes, uvs, mutex_uvs, weights, mutex_weights
    )
    grid_graph.relabel_to_seeds(segmentation)
    segmentation = segmentation.reshape(shape)
    if mask is not None:
        segmentation[np.logical_not(mask)] = 0

    return segmentation


def run(
        pred_file,
        pred_dataset,
        roi,
        downsample,
        normalize_preds,
        neighborhood,
        stride,
        randomize_strides,
        algorithm,
        mask_thresh):

    # load
    pred = daisy.open_ds(pred_file,pred_dataset)

    if roi is not None:
        roi = daisy.Roi(pred.roi.offset+daisy.Coordinate(roi[0]),roi[1])
    else:
        roi = pred.roi

    pred = pred.to_ndarray(roi)
    
    # normalize
    pred = (pred / np.max(pred)).astype(np.float32)
    
    # normalize channel-wise
    if normalize_preds:
        for c in range(len(pred)):
            
            max_v = np.max(pred[c])
            min_v = np.min(pred[c])

            if max_v != min_v:
                pred[c] = (pred[c] - min_v)/(max_v - min_v)
            else:
                pred[c] = np.ones_like(pred[c])
    
    # prepare
    neighborhood = [tuple(x) for x in neighborhood]
    
    if mask_thresh > 0.0 or algorithm == "seeded":
        
        # TO-DO: be able to manually feed seeds array.
        
        mean_pred = 0.5 * (pred[1] + pred[2])
        depth = mean_pred.shape[0]

        if mask_thresh > 0.0:
            mask = np.zeros(mean_pred.shape, dtype=bool)
        
        if algorithm == "seeded":
            seeds = np.zeros(mean_pred.shape, dtype=np.uint64)

        for z in range(depth):

            boundary_mask = mean_pred[z] > mask_thresh * np.max(pred)
            boundary_distances = distance_transform_edt(boundary_mask)
            if mask_thresh > 0.0:
                mask[z] = boundary_mask

            if algorithm == "seeded":
                _,_,seeds[z] = watershed_from_boundary_distance(
                    boundary_distances,
                    return_seeds=True,
                )
    
    if mask_thresh == 0.0:
        mask = None

    if "seeded" in algorithm:
        seeds = seeds if "wo" not in algorithm else None   
        
        seg = seeded_mutex_watershed(
            seeds=seeds,
            affs=pred,
            offsets=neighborhood,
            mask=mask,
            stride=stride,
            randomize_strides=randomize_strides)
    
    else:
        seg = mutex_watershed(
            pred,
            offsets=neighborhood,
            stride=stride,
            algorithm=algorithm,
            mask=mask,
            randomize_strides=randomize_strides)

    if downsample > 1:
            seg = rescale(segmentation.copy(), [1,downsampling[1],downsampling[1]], order=0)
    
    return seg
