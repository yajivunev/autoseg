import os
import sys
import json
import logging
import gunpowder as gp
import numpy as np

logging.basicConfig(level=logging.INFO)


def train(
        iterations,
        save_every,
        sources,
        model_path,
        config_path=None,
        pipeline_path=None,
        downsample=False,
        min_masked=0.5,
        probabilities=None,
        pre_cache=None,
        checkpoint_basename="model",
        log_dir=None,
        snapshots_dir=None):

    """ Runs training of given model on given sources. 

    Args:

        iterations (``int``):
            
            The number of training iterations.

        save_every (``int``):

            Saves model checkpoints and snapshots (if requested) every save_every iterations.

        sources (``list`` of ``dicts``):

            Each dict in the list is a source. A source of the form:
                
                source_i = {
                    "array_A": [zarr_container, dataset_name_A],
                    "array_B": [zarr_container, dataset_name_B],
                    ...
                    }

            Each source dict must provide for all the arrays in the pipeline. 
            Each source must have the same number of elements. 
            Refer to your model's pipeline for arrays required in the sources.

        model_path (``string``):

            Path to "model.py" which contains the Torch model class used
            for training.
        
        config_path (``string``, optional):

            Path to "config.json" which contains the paramters for the model
            construction, training, inference and post-processing. 

        pipeline_path (``string``, optional):

            Path to "pipeline.py" which containts the training pipeline constructor.

        checkpoint_basename (``string``, optional):

            The basename used for checkpoint files. Defaults to ``model``.

            Path to an existing model checkpoint folder to start training with the 
            latest checkpoint's weights loaded.
            Default is no checkpoint, i.e,  starting from scratch.
        
        downsample (``bool`` or ``int``, optional):

            Downsampling factor to downsample volumes in XY before training.
            Default is False. True will result in downsampling factor being 
            automatically calulated by doing the following:

                factor = int(
                    round(
                        model_default_voxel_size/given_source_voxel_size
                        ))
            
            This may result in total ROI being slightly underscanned or
            overscanned with respect to the true total ROI.

        min_masked (``float``, optional):

            Float between 0 and 1 which represents the fraction of the training example with labels.

        probabilities (``list`` of ``floats``, optinal):

            List of floats of the same length as `sources` that sum to 1. Each float is the probability
            a training example will be requested from the correponding source. 
            Default is equal probability.

        pre_cache (``tuple`` of ``ints``, optional):

            Tuple of ints containing the number of workers and cache_workers to be used to pre-load
            training batches using multi-processing before they are sent to the model. 
            Default is None, which is no pre-caching. 

        log_dir (``string``, optional):

            Directory for saving tensorboard summaries.

        snapshots_dir (``string``, optional):

            The directory to save the snapshots. Will be created, if it does
            not exist. Default is None.
    """
    
    # import model from model_path
    sys.path.append(os.path.dirname(model_path))
    from model import Model

    # load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path),"config.json")
    
    with open(config_path,"r") as f:
        config = json.load(f)
 
    model = Model(config_path)
    model.train()

    # load pipeline
    if pipeline_path is None:
        pass
    else:
        sys.path.append(os.path.dirname(pipeline_path))
    
    from pipeline import Pipeline

    pipeline = Pipeline(config_path)
    train_pipeline, train_request = pipeline.get_train_pipeline(
            model,
            sources,
            downsample,
            min_masked,
            probabilities,
            save_every,
            checkpoint_basename,
            pre_cache,
            log_dir,
            snapshots_dir)
    
    # train
    with gp.build(train_pipeline):
        for i in range(iterations):
            batch = train_pipeline.request_batch(train_request)


if __name__ == "__main__":

    iterations = 10001
    save_every = 1000
    sources = [{
                    "raw": ("/home/vijay/science/models/membrane_mtlsd_Rat34/dsnyj_crop.zarr","raw"),
                    "labels": ("test.zarr","seg_0.55_filtered_filtered"),
                    "unlabelled": ("test.zarr","unlabelled")
                }]
    model_path = "/home/vijay/science/autoseg/autoseg/models/membrane/mtlsd_2.5d_unet/model.py"

    train(
        iterations,
        save_every,
        sources,
        model_path,
        pre_cache=(10,40),
        downsample=False,
        snapshots_dir="snapshots",
        log_dir="log")
