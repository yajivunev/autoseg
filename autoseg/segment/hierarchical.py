import sys
import numpy as np
import daisy
from funlib.persistence import open_ds, prepare_ds
import zarr

from skimage.transform import rescale
import waterz

from watershed import watershed_from_affinities


waterz_merge_function = {
    'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
    'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
    'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
    'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
    'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
    'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
    'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
    'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
    'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
    'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
    'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
}

max_thresh = 1.0
step = 1/20

thresholds = [round(x,2) for x in np.arange(0,max_thresh,step)]


def run(
        pred_file,
        pred_dataset,
        roi=None,
        downsample=1,
        normalize_preds=False,
        min_seed_distance=10,
        merge_function="mean",
        thresholds=thresholds):

    # load
    pred = open_ds(pred_file,pred_dataset)
    #voxel_size = pred.voxel_size
    
    if roi is not None:
        roi = daisy.Roi(pred.roi.offset+daisy.Coordinate(roi[0]),roi[1])
    else:
        roi = pred.roi

    pred = pred.to_ndarray(roi)

    # first three channels are direct neighbor affs
    if len(pred) > 3:
        pred = pred[:3]
    
    # downsample in XY
    if downsample > 1:
        
        pred = rescale(
            pred,
            [1,1,1/downsample,1/downsample],
            anti_aliasing=True,
            order=1)

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
   
    # watershed
    fragments = watershed_from_affinities(
        pred,
        fragments_in_xy=True,
        min_seed_distance=min_seed_distance)[0]
    
    # agglomerate
    generator = waterz.agglomerate(
            pred,
            thresholds=thresholds,
            fragments=fragments.copy(),
            scoring_function=waterz_merge_function[merge_function])

    # write
    f = zarr.open(pred_file,"a")
    
    for threshold,segmentation in zip(thresholds,generator):
       
        if downsample > 1:
            seg = rescale(segmentation.copy(), [1, downsample, downsample], order=0)
        else:
            seg = segmentation.copy()
        
        print(f"Writing segmentation at {threshold} to {pred_file}")
        f[f"seg_{threshold}"] = seg
        f[f"seg_{threshold}"].attrs["offset"] = roi.get_offset()
        f[f"seg_{threshold}"].attrs["resolution"] = [int(x/y) for x,y in zip(roi.get_shape(),seg.shape)]
        
    return roi


if __name__ == "__main__":

    pred_file = sys.argv[1]
    pred_dataset = sys.argv[2]
    roi = None

    roi = run(
        pred_file,
        pred_dataset,
        thresholds=[0.5])
