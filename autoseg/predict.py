import os
import sys
import gunpowder as gp
import numpy as np
import daisy
from funlib.persistence import open_ds, prepare_ds

def predict(
        sources,
        roi,
        model_path,
        checkpoint_path,
        out_file,
        write="all"):

    """
    Do inference on raw_file/raw_dataset using trained model.

    Args:
        
        sources (``list`` of ``tuple`` of two ``strings``):
            
            List of zarr sources. 
            sources = [
                (source1_container_path, source1_dataset),
                (source2_container_path, source2_dataset),
                ...]

        roi (``tuple`` of two ``tuples`` of three ``ints``):

            Region of intereset of the source_dataset to use for
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

            To choose which of the output datasets to write to zarr container.

            "all" : Writes all arrays your model outputs (default).
            "affs" : Writes only affinities, if your model outputs it.
            "lsds" : Write only local shape descriptors, if your model outputs it.
            "mask" : Writes only mask, if your model outputs it.
            None : Writes nothing. Returns the output arrays in memory. 
    """
   
    # import model from model_path and make instance
    sys.path.append(os.path.dirname(model_path))
    from model import model
    from model import inference_array_keys as keys

    model.eval()

    # get checkpoint iteration
    iteration = checkpoint_path.split('_')[-1]

    # I/O shapes and sizes
    # TO-DO: get I/O shapes from model_path?
    increase = gp.Coordinate([32, 8*12, 8*12])
    input_shape = gp.Coordinate([24, 196, 196]) + increase
    output_shape = gp.Coordinate([8, 104, 104]) + increase

    # nm
    # TO-DO: ask for voxel_size; 
    # if raw's voxel_size <= given voxel_size, Downsample. else Error.
    voxel_size = gp.Coordinate([50, 8, 8])
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    context = (input_size - output_size) / 2

    # get input datasets, output datasets, array keys from model
    in_keys = []
    out_keys = []
    out_ds_names = []

    for in_key in keys[0].keys():
        in_keys.append(gp.ArrayKey(in_key))

    for out_key,num_channels in keys[1].items():
        out_keys.append(gp.ArrayKey(out_key))
        if write=="all" or out_key.lower().split('_')[-1] in write: 
            out_ds_names.append((f"{out_key.lower()}_{iteration}",num_channels,out_key))

    # get ROI, grow input_roi by context
    if roi is None:
        ds = open_ds(sources[0][0],sources[0][1])
        total_output_roi = gp.Roi(gp.Coordinate(ds.roi.get_offset()),gp.Coordinate(ds.roi.get_shape()))
        total_input_roi = total_output_roi.grow(context,context)

    else:
        total_output_roi = gp.Roi(gp.Coordinate(roi[0]),gp.Coordinate(roi[1]))
        total_input_roi = total_output_roi.grow(context,context)

    print(total_input_roi,total_output_roi)

    # prepare output zarr datasets
    if out_ds_names != []:
        for out_ds_name,num_channels,_ in out_ds_names:

            prepare_ds(
                out_file,
                out_ds_name,
                daisy.Roi(
                    total_output_roi.get_offset(),
                    total_output_roi.get_shape()
                ),
                voxel_size,
                np.uint8,
                write_size=output_size,
                compressor={'id': 'blosc'},
                delete=True,
                num_channels=num_channels)

    # add specs to scan_request
    scan_request = gp.BatchRequest()
    
    for in_key in in_keys:
        scan_request.add(in_key, input_size)
    
    for out_key in out_keys:
        scan_request.add(out_key, output_size)

    # get zarr sources
    assert len(sources) == len(in_keys), "Too many sources, probably"
    
    # extra unsqueeze for single-channel image arrays
    extra_unsqueeze = [gp.Unsqueeze([in_key]) 
            if keys[0][str(in_key)] == 1 
            else gp.Unsqueeze([in_key])+gp.Squeeze([in_key]) 
            for in_key in in_keys
            ]

    # make zarr sources
    sources = tuple(
            gp.ZarrSource(
                sources[i][0],
            {
                in_keys[i]: sources[i][1]
            },
            {
                in_keys[i]: gp.ArraySpec(interpolatable=True)
            }) +
            gp.Normalize(in_keys[i]) +
            gp.Pad(in_keys[i], None) +
            gp.IntensityScaleShift(in_keys[i], 2,-1) +
            gp.Unsqueeze([in_keys[i]]) + 
            extra_unsqueeze[i]
            for i,source in enumerate(sources))

    if len(sources) > 1:
        sources += gp.MergeProvider()

    else:
        sources = sources[0]
    
    # make pipeline
    pipeline = sources

    pipeline += gp.torch.Predict(
            model,
            checkpoint=checkpoint_path,
            inputs = {f"input_{str(k).lower()}":k for k in in_keys},
            outputs = {k:v for k,v in enumerate(out_keys)},
            )

    # remove batch dimension for writing
    pipeline += gp.Squeeze(out_keys)
   
    # uint8 scaling
    for out_key in out_keys:
        pipeline += gp.IntensityScaleShift(out_key, 255, 0)

    # return arrays if nothing is being written
    return_arrays = False

    if write is not None or write != False or write == []:
        dataset_names = {gp.ArrayKey(k):v for v,_,k in out_ds_names}
        print(f"Writing to {out_file}: {dataset_names}")
        pipeline += gp.ZarrWrite(
                dataset_names=dataset_names,
                store=out_file)
    else:
        return_arrays = True

    pipeline += gp.Scan(scan_request)

    predict_request = gp.BatchRequest()
    
    for in_key in in_keys:
        predict_request[in_key] = total_input_roi
    for out_key in out_keys:
        predict_request[out_key] = total_output_roi
    
    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    if return_arrays:
        arrays = [batch[out_key].data for out_key in out_keys] 
        return *arrays, total_output_roi.get_begin()
    else:
        return total_output_roi.get_begin()


if __name__ == "__main__":

    sources = [(sys.argv[1],"raw")]
    roi = None#((500,4000,4000),(1500,8000,8000))
    model_path = "models/membrane/mtlsd_2.5d_unet/model.py"
    checkpoint_path = sys.argv[2]
    out_:w
    file = "test.zarr"

    predict(
        sources,
        roi,
        model_path,
        checkpoint_path,
        out_file,
        write="all")
