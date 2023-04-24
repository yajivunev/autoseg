import os
import sys
import gunpowder as gp
import numpy as np
import daisy
from funlib.persistence import prepare_ds


def predict(
        raw_file,
        raw_dataset,
        roi,
        model_path,
        checkpoint_path,
        out_file,
        write="all"):

    """
    Do inference on raw_file/raw_dataset using trained model.

    Args:
        
        raw_file (``string``):

            Path to zarr container containing raw_dataset.

        raw_dataset (``string``):

            Name of the 3D image dataset in raw_file.

        roi (``tuple`` of two ``tuples`` of ``ints``):

            Region of intereset of the raw_dataset to use for
            inference in world units (not voxels!) 

            roi = ((offset_z, offset_y, offset_x), (size_z, size_y, size_x))

        model_path (``string``):

            Path to "model.py" which contains the Torch model instance used
            for inference.

        checkpoint_path (``string``):

            Path to model checkpoint. 

        out_file (``string``):

            Path to output zarr container.

        write (``string``, optional):

            To choose which output datasets to write to zarr container.

            "all" : Writes all datasets (default).
            "affs" : Writes only affinities.
            "lsds" : Write only local shape descriptors.
            None : Writes nothing. Returns the output arrays in memory. 
    """
   
    # import model from model_path and make instance
    sys.path.append(os.path.dirname(model_path))
    from model import model

    model.eval()

    # get checkpoint iteration
    iteration = checkpoint_path.split('_')[-1]

    # I/O shapes and sizes
    # TO-DO: get I/O shapes from model_path?
    increase = gp.Coordinate([16, 8*16, 8*16])
    input_shape = gp.Coordinate([24, 196, 196]) + increase
    output_shape = gp.Coordinate([8, 104, 104]) + increase

    # nm
    # TO-DO: ask for voxel_size; 
    # if raw's voxel_size <= given voxel_size, Downsample. else Error.
    voxel_size = gp.Coordinate([50, 8, 8])
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    context = (input_size - output_size) // 2

    # TO-DO: use model path to get gp.ArrayKeys and request specs
    lsds_out_ds = f"lsds_{iteration}"
    affs_out_ds = f"affs_{iteration}"

    raw = gp.ArrayKey('RAW')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)

    source = gp.ZarrSource(
                raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(interpolatable=True)
            })

    if roi is None:
        with gp.build(source):
            total_output_roi = source.spec[raw].roi
            total_input_roi = source.spec[raw].roi.grow(context,context)

    else:
        total_output_roi = gp.Roi(gp.Coordinate(roi[0]),gp.Coordinate(roi[1]))
        total_input_roi = total_output_roi.grow(context,context)

    # prepare output zarr datasets
    if write=="all" or write=="lsds":
        prepare_ds(
                out_file,
                lsds_out_ds,
                daisy.Roi(
                    total_output_roi.get_offset(),
                    total_output_roi.get_shape()
                ),
                voxel_size,
                np.uint8,
                write_size=output_size,
                compressor={'id': 'blosc'},
                num_channels=10)
    
    if write=="all" or write=="affs":
        prepare_ds(
                out_file,
                affs_out_ds,
                daisy.Roi(
                    total_output_roi.get_offset(),
                    total_output_roi.get_shape()
                ),
                voxel_size,
                np.uint8,
                write_size=output_size,
                compressor={'id': 'blosc'},
                num_channels=3)

    predict = gp.torch.Predict(
            model,
            checkpoint=checkpoint_path,
            inputs = {
                'input': raw
            },
            outputs = {
                0: pred_lsds,
                1: pred_affs,
            })

    scan = gp.Scan(scan_request)

    return_arrays = False

    if write=="all":
        write = gp.Squeeze([pred_lsds,pred_affs])
        write += gp.IntensityScaleShift(pred_lsds, 255, 0)
        write += gp.IntensityScaleShift(pred_affs, 255, 0)
        write += gp.ZarrWrite(
                dataset_names={
                    pred_lsds: lsds_out_ds,
                    pred_affs: affs_out_ds
                },
                store=out_file)
    
    elif write=="affs":
        write = gp.Squeeze([pred_affs])
        write += gp.IntensityScaleShift(pred_affs, 255, 0)
        write += gp.ZarrWrite(
                dataset_names={
                    pred_affs: affs_out_ds
                },
                store=out_file)
    
    elif write=="lsds":
        write += gp.Squeeze([pred_lsds])
        write += gp.IntensityScaleShift(pred_lsds, 255, 0)
        write += gp.ZarrWrite(
                dataset_names={
                    pred_lsds: lsds_out_ds
                },
                store=out_file)
    
    else:
        write = gp.Squeeze([pred_lsds,pred_affs])
        write += gp.IntensityScaleShift(pred_lsds, 255, 0)
        write += gp.IntensityScaleShift(pred_affs, 255, 0)
        return_arrays = True

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.IntensityScaleShift(raw, 2,-1) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            write+
            scan)

    predict_request = gp.BatchRequest()
    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi
    predict_request[pred_affs] = total_output_roi
    
    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    if return_arrays:
        return batch[pred_affs].data, batch[pred_lsds].data, total_output_roi.get_begin()
    else:
        return total_output_roi.get_begin()


if __name__ == "__main__":

    raw_file = sys.argv[1]
    raw_dataset = "raw"
    roi = None#((500,4000,4000),(1500,8000,8000))
    #roi = ((50, 4160, 5008),(260*50,763*8,1004*8))
    model_path = "models/membrane/mtlsd_2.5d_unet/model.py"
    checkpoint_path = sys.argv[2]
    out_file = raw_file#"test.zarr"

    predict(
        raw_file,
        raw_dataset,
        roi,
        model_path,
        checkpoint_path,
        out_file,
        write="affs")
