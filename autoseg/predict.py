import os
import sys
import json
import gunpowder as gp
import numpy as np
import daisy
from funlib.persistence import open_ds, prepare_ds

def predict(
        sources,
        out_file,
        checkpoint_path,
        model_path,
        config_path=None,
        roi=None,
        write="all",
        increase=None,
        downsample=False):

    """
    Do inference on raw_file/raw_dataset using trained model.

    Args:
        
        sources (``list`` of ``tuple`` of two ``strings``):
            
            List of zarr sources. 
            sources = [
                (source1_container_path, source1_dataset),
                (source2_container_path, source2_dataset),
                ...]

        out_file (``string``):

            Path to output zarr container to write inference outputs.

        checkpoint_path (``string``):

            Path to model checkpoint. 

        model_path (``string``):

            Path to "model.py" which contains the Torch model class used
            for inference.
        
        config_path (``string``, optional):

            Path to "config.json" which contains the paramters for the model
            construction, training, inference and post-processing.

        roi (``tuple`` of two ``tuples`` of three ``ints``, optional):

            Region of intereset of the source_dataset to use for
            inference in world units (not voxels!) 

            roi = ((offset_z, offset_y, offset_x), (size_z, size_y, size_x))

        write (``string``, optional):

            To choose which of the output datasets to write to zarr container.

            "all" : Writes all arrays your model outputs (default).
            "affs" : Writes only affinities, if your model outputs it.
            "lsds" : Write only local shape descriptors, if your model outputs it.
            "mask" : Writes only mask, if your model outputs it.
            None : Writes nothing. Returns the output arrays in memory. 

        increase (``list`` or ``tuple`` of ``ints``, optional):

            Number of voxels to add to each dimension of the model's input and output shapes.
            Default value is [0,] * number of dimensions.

        downsample (``bool`` or ``int``, optional):

            Downsampling factor to downsample volumes in XY before prediction.
            Default is False. True will result in downsampling factor being 
            automatically calulated by doing the following:

                factor = int(
                    round(
                        model_default_voxel_size/given_source_voxel_size
                        ))
            
            This may result in total_output_roi being slightly underscanned or
            overscanned with respect to the true total output ROI.

    """
   
    # import model from model_path
    sys.path.append(os.path.dirname(model_path))
    from model import Model

    # load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path),"config.json")
    
    with open(config_path,"r") as f:
        config = json.load(f)

        # TO-DO: lazy. make prettier
        keys = config["predict"]["keys"]
        output_shapes = config["model"]["output_shapes"] 
        input_shape = config["model"]["input_shape"] 
        output_shape = config["model"]["output_shape"] 
        input_shape = config["model"]["input_shape"] 
        default_voxel_size = config["model"]["default_voxel_size"] 
    
    model = Model(config_path)
    model.eval()

    # get section number of source if 2D sources
    if len(input_shape) == 2:
        section = "/" + sources[0][1].split('/')[-1]
    else:
        section = ""

    # get checkpoint iteration
    iteration = checkpoint_path.split('_')[-1]

    # get input datasets, output datasets, array keys from model
    in_keys = []
    fr_in_keys = [] # full-resolution
    out_keys = []
    out_ds_names = []

    for in_key in keys["input"].keys():
        in_keys.append(gp.ArrayKey(in_key))
        fr_in_keys.append(gp.ArrayKey(in_key + "_FR"))

    for out_key,num_channels in keys["output"].items():
        out_keys.append(gp.ArrayKey(out_key))
        if write=="all" or out_key.lower().split('_')[-1] in write: 
            out_ds_names.append((f"{out_key.lower()}_{iteration}{section}",num_channels,out_key))
    
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

    # get ROI, grow input_roi by context
    if roi is None:
        ds = open_ds(sources[0][0],sources[0][1])
        total_output_roi = gp.Roi(
                gp.Coordinate(ds.roi.get_offset()),
                gp.Coordinate(ds.roi.get_shape()))
        total_input_roi = total_output_roi.grow(context,context)

    else:
        total_output_roi = gp.Roi(gp.Coordinate(roi[0]),gp.Coordinate(roi[1]))
        total_input_roi = total_output_roi.grow(context,context)

    for i in range(len(voxel_size)):
        assert total_output_roi.get_shape()[i]/voxel_size[i] >= output_shape[i], \
            "total output (write) ROI cannot be smaller than model's output shape" 

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
            if keys["input"][str(in_key)] == 1 
            else gp.Unsqueeze([in_key])+gp.Squeeze([in_key]) 
            for in_key in in_keys
            ]

    # make zarr sources
    sources = tuple(
            gp.ZarrSource(
                sources[i][0],
            {
                fr_in_keys[i]: sources[i][1]
            },
            {
                fr_in_keys[i]: gp.ArraySpec(interpolatable=True)
            }) +
            gp.Normalize(fr_in_keys[i]) +
            gp.Pad(fr_in_keys[i], None) +
            gp.DownSample(fr_in_keys[i],downsample_factors,in_keys[i]) +
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
        print(f"Writing to {out_file}: {dataset_names} with voxel_size={voxel_size}")
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

    sources = [(sys.argv[1],sys.argv[2])]
    roi = None#((500,4000,4000),(1000,4000,4000))
    #model_path = "models/membrane/mtlsd_2.5d_unet/model.py"
    model_path = "models/membrane/lsd_2d_unet/model.py"
    checkpoint_path = sys.argv[3]
    out_file = "test.zarr"
    #increase = [8,8*10,8*10]
    increase = [8*10,8*10]

    predict(
        sources,
        out_file,
        checkpoint_path,
        model_path,
        config_path=None,
        roi=roi,
        write="all",
        increase=increase,
        downsample=False)
