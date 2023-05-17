import os
import sys
import math
import json
import numpy as np
import gunpowder as gp
from funlib.persistence import open_ds, prepare_ds
from lsd.train.gp import AddLocalShapeDescriptor

script_dir = os.path.dirname(__file__)

from autoseg.utils import (
        SmoothArray, 
        RandomNoiseAugment,
        ZerosSource,
        CreateLabels,
        CustomLSDs
)


class Pipeline():

    def __init__(self, config_path):

        self.config_path = config_path

    def load_config(self, mode):

        with open(self.config_path,"r") as f: 
            config = json.load(f)

        for k,v in config["model"].items():

            if type(v) == str:
                value = f'"{v}"'
            elif k == "params":
                pass
            else:
                value = str(v)

            exec(f'self.{k} = {value}')

        if mode == "train":
            
            for k,v in config["train"].items():
                
                if type(v) == str:
                    value = f'"{v}"'
                else:
                    value = str(v)

                exec(f'self.{k} = {value}')
        
        if mode == "predict":
            
            for k,v in config["predict"].items():
                
                if type(v) == str:
                    value = f'"{v}"'
                else:
                    value = str(v)

                exec(f'self.{k} = {value}')

    def get_predict_pipeline(
            self,
            model,
            sources,
            checkpoint_path,
            increase,
            downsample,
            roi,
            out_roi,
            write,
            out_file):

        self.load_config("predict")
        model.eval()

        # get section number of source if 2D sources
        if len(self.input_shape) == 2:
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

        for in_key in self.keys["input"].keys():
            in_keys.append(gp.ArrayKey(in_key))
            fr_in_keys.append(gp.ArrayKey(in_key + "_FR"))

        for out_key,num_channels in self.keys["output"].items():
            out_keys.append(gp.ArrayKey(out_key))
            if write=="all" or out_key.lower().split('_')[-1] in write: 
                out_ds_names.append((f"{out_key.lower()}_{iteration}{section}",num_channels,out_key))
                
        # I/O shapes and sizes
        if increase is not None:
            increase = gp.Coordinate(increase)
        else:
            increase = gp.Coordinate([0,]*len(self.input_shape))

        input_shape = gp.Coordinate(self.input_shape) + increase
        output_shape = gp.Coordinate(self.output_shape) + increase

        voxel_size = open_ds(sources[0][0],sources[0][1]).voxel_size
        default_voxel_size = gp.Coordinate(self.default_voxel_size)

        # XY downsample factor
        if downsample==True:
            downsample = int(round(default_voxel_size[-1]/voxel_size[-1]))
        elif type(downsample) == int:
            pass
        else:
            downsample = 1

        downsample_factors = (1,) * (len(self.input_shape) - 2) + (downsample,downsample)
        voxel_size = voxel_size * gp.Coordinate(downsample_factors) if downsample > 1 else voxel_size
        
        # world units (nm)
        input_size = input_shape * voxel_size
        output_size = output_shape * voxel_size
        context = (input_size - output_size) // 2

        # get ROI, grow input_roi by context if full
        if out_roi == "full":
            if roi is None:

                ds = open_ds(sources[0][0],sources[0][1])

                total_output_roi = gp.Roi(
                        gp.Coordinate(ds.roi.get_offset()),
                        gp.Coordinate(ds.roi.get_shape()))

                total_input_roi = total_output_roi.grow(context, context)

            else:

                total_output_roi = gp.Roi(gp.Coordinate(roi[0]), gp.Coordinate(roi[1]))
                total_input_roi = total_output_roi.grow(context, context)
        else:
            if roi is None:

                ds = open_ds(sources[0][0],sources[0][1])

                total_input_roi = gp.Roi(
                        gp.Coordinate(ds.roi.get_offset()),
                        gp.Coordinate(ds.roi.get_shape()))

                total_output_roi = total_input_roi.grow(-context, -context)

            else:

                total_input_roi = gp.Roi(gp.Coordinate(roi[0]), gp.Coordinate(roi[1]))
                total_output_roi = total_input_roi.grow(-context, -context)
  

        for i in range(len(voxel_size)):
            assert total_output_roi.get_shape()[i]/voxel_size[i] >= output_shape[i], \
                f"total output (write) ROI cannot be smaller than model's output shape, \ni: {i}\ntotal_output_roi: {total_output_roi.get_shape()[i]}, \noutput_shape: {output_shape[i]}, \nvoxel size: {voxel_size[i]}" 
 
        # prepare output zarr datasets
        if out_ds_names != []:
            for out_ds_name,num_channels,_ in out_ds_names:

                prepare_ds(
                    out_file,
                    out_ds_name,
                    total_output_roi,
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
                if self.keys["input"][str(in_key)] == 1 
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
        
        # write
        if write is not None or write != False:
            dataset_names = {gp.ArrayKey(k):v for v,_,k in out_ds_names}
            print(f"Writing to {out_file}: {dataset_names} with voxel_size={voxel_size}")
            pipeline += gp.ZarrWrite(
                    dataset_names=dataset_names,
                    store=out_file)

        pipeline += gp.Scan(scan_request)

        # predict request
        request = gp.BatchRequest()
        
        for in_key in in_keys:
            request[in_key] = total_input_roi
        for out_key in out_keys:
            request[out_key] = total_output_roi

        return pipeline, request, [item[0] for item in out_ds_names]
    
    def _make_train_augmentation_pipeline(self, raw, source):

        augs = self.augmentations

        if 'elastic' in augs.keys():
            source = source + gp.ElasticAugment(rotation_interval=[0, math.pi/2], **augs["elastic"])

        if 'simple' in augs.keys():
            source = source + gp.SimpleAugment(**augs["simple"])

        if 'noise' in augs.keys():
            source = source + RandomNoiseAugment(raw)

        if 'intensity' in augs.keys():
            source = source + gp.IntensityAugment(raw, **augs["intensity"])

        if 'blur' in augs.keys():
            source = source + SmoothArray(raw, **augs["blur"])
        
        return source


    def get_train_pipeline(
            self,
            model,
            sources,
            downsample,
            min_masked,
            probabilities,
            save_every,
            checkpoint_basename,
            pre_cache,
            log_dir,
            snapshots_dir):

        self.load_config("train")

        # model, loss, optimizer
        model.train()
        loss = model.return_loss() 
        optimizer = model.return_optimizer()(model.parameters(),**self.optimizer["params"])

        # array keys 
        zeros = gp.ArrayKey("ZEROS")
        gt_lsds = gp.ArrayKey("GT_LSDS")
        gt_affs = gp.ArrayKey("GT_AFFS")
        pred_affs = gp.ArrayKey("PRED_AFFS")
        affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

        # I/O shapes and sizes
        input_shape = gp.Coordinate(self.input_shape)
        output_shape = gp.Coordinate(self.output_shape)

        # assuming all sources have same voxel_size -- they will be set to first source's if not.
        default_voxel_size = gp.Coordinate(self.default_voxel_size)
        voxel_size = gp.Coordinate(self.default_voxel_size)

        # XY downsample factor
        if downsample==True:
            downsample = int(round(default_voxel_size[-1]/voxel_size[-1]))
        elif type(downsample) == int:
            pass
        else:
            downsample = 1

        downsample_factors = (1,) * (len(input_shape) - 2) + (downsample,downsample)
        voxel_size = voxel_size * gp.Coordinate(downsample_factors) if downsample > 1 else voxel_size
       
        anisotropy = int((voxel_size[0] / voxel_size[1]) - 1) # 0 is isotropic

        # world units (nm)
        input_size = input_shape * voxel_size
        output_size = output_shape * voxel_size
        context = (input_size - output_size) // 2

        # add specs to request
        request = gp.BatchRequest()
        request.add(zeros, input_size)
        request.add(gt_lsds, input_size)
        request.add(gt_affs, output_size)
        request.add(pred_affs, output_size)
        request.add(affs_weights, output_size)

        # make source
        source = ZerosSource(
            {
                zeros: "zeros",  # just a zeros dataset, since we need a source
            },
            shape=input_shape,
            array_specs={
                zeros: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
            },
        )

        source += gp.Pad(zeros, None)

        # make pipeline
        pipeline = sources

        # create fake 3d labels
        pipeline += CreateLabels(zeros, anisotropy)

        # add augmentations
        pipeline = self._make_train_augmentation_pipeline(zeros, pipeline)

        # add learning targets
        for node, params in self.nodes.items():

            if "add_custom_lsds" in node:
                pipeline += CustomLSDs( 
                    zeros,
                    gt_lsds,
                    unlabelled=unlabelled,
                    lsds_mask=lsds_weights,
                    downsample=1,
                    **params)
            
            if 'noise' in node:
                pipeline += RandomNoiseAugment(gt_lsds)

            if 'intensity' in node:
                pipeline += gp.IntensityAugment(gt_lsds, **augs["intensity"])

            if 'blur' in node:
                pipeline += SmoothArray(gt_lsds, **augs["blur"])

            if "grow_boundary" in node:
                pipeline += gp.GrowBoundary(
                    zeros,
                    **params)
    
            if "add_affs" in node:
                pipeline += gp.AddAffinities(
                    affinity_neighborhood=params["neighborhood"],
                    labels=zeros,
                    affinities=gt_affs,
                    dtype=np.float32)
            
            if "balance_labels" in node:
                pipeline += gp.BalanceLabels(
                    gt_affs,
                    affs_weights)

        # add remaining nodes
        pipeline += gp.Stack(1)

        if pre_cache is not None:
            pipeline += gp.PreCache(num_workers=pre_cache[0], cache_size=pre_cache[1])

        pipeline += gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={
                'input_lsds': gt_lsds
            },
            outputs={
                0: pred_affs,
            },
            loss_inputs={
                0: pred_affs,
                1: gt_affs,
                2: affs_weights,
            },
            save_every=save_every,
            log_dir=log_dir,
            checkpoint_basename=checkpoint_basename)

        pipeline += gp.Squeeze([gt_lsds,gt_affs,pred_affs,affs_weights])

        if snapshots_dir is not None:
            pipeline += gp.Snapshot(
                    dataset_names={
                        gt_lsds: 'gt_lsds',
                        gt_affs: 'gt_affs',
                        pred_affs: 'pred_affs',
                        affs_weights: 'affs_weights',
                    },
                    dataset_dtypes={
                        gt_affs: np.float32
                    },
                    output_filename='batch_{iteration}.zarr',
                    output_dir=snapshots_dir,
                    every=save_every
            )

        return pipeline, request
