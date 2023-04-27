import os
import sys
import json
import gunpowder as gp

def predict(
        sources,
        out_file,
        checkpoint_path,
        model_path,
        config_path=None,
        pipeline_path=None,
        roi=None,
        write="all",
        increase=None,
        downsample=False,
        return_arrays=False):

    """
    Do inference on raw_file/raw_dataset using trained model.

    Args:
        
        sources (``list`` of ``tuple`` of two ``strings``):
            
            List of zarr sources. 
            sources = [
                (source1_container_path, source1_dataset),
                (source2_container_path, source2_dataset),
                ...]

            Multiple sources should only be used if model requires multiple inputs.

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

        pipeline_path (``string``, optional):

            Path to "pipeline.py" which containts the inference pipeline constructor.

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

        return_arrays (``bool``, optional):

            Return predicted numpy arrays.

    """
   
    # import model from model_path
    sys.path.append(os.path.dirname(model_path))
    from model import Model

    # load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path),"config.json")
    
    with open(config_path,"r") as f:
        config = json.load(f)
        keys = config["predict"]["keys"]
    
    model = Model(config_path)
    model.eval()
    
    # load pipeline
    if pipeline_path is None:
        pass
    else:
        sys.path.append(os.path.dirname(pipeline_path))
    
    from pipeline import Pipeline
    pipeline = Pipeline(config_path)

    pipeline, request, output_datasets = pipeline.get_predict_pipeline(
            model,
            sources,
            checkpoint_path,
            increase,
            downsample,
            roi,
            write,
            out_file)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    if return_arrays:
        arrays = [batch[gp.ArrayKey(key)].data for key in keys["output"].keys()] 
        return *arrays, output_datasets
    else:
        return output_datasets


if __name__ == "__main__":

    sources = [(sys.argv[1],sys.argv[2])]
    roi = None#((500,4000,4000),(1000,4000,4000))
    model_path = "models/membrane/mtlsd_2.5d_unet/model.py"
    checkpoint_path = sys.argv[3]
    out_file = "test.zarr"
    increase = [8,8*10,8*10]

    predict(
        sources,
        out_file,
        checkpoint_path,
        model_path,
        roi=roi,
        write="affs",
        increase=increase)
