# autoseg
modules and scripts for machine learning on EM

## installation
```
conda create -n autoseg python=3.9
conda activate autoseg
conda install pytorch pytorch-cuda=11.7 boost jupyter -c pytorch -c nvidia
```
```
pip install cython zarr matplotlib mahotas
pip install git+https://github.com/funkelab/funlib.geometry.git
pip install git+https://github.com/funkelab/funlib.persistence.git
pip install git+https://github.com/funkelab/daisy.git
pip install git+https://github.com/funkey/gunpowder.git
pip install git+https://github.com/funkelab/funlib.math.git
pip install git+https://github.com/funkelab/funlib.evaluate.git
pip install git+https://github.com/funkelab/funlib.learn.torch.git
pip install git+https://github.com/htem/waterz@hms
pip install git+https://github.com/funkelab/funlib.segment.git
pip install git+https://github.com/funkelab/lsd.git
pip install neuroglancer cloud-volume
pip install tensorboard tensorboardx
pip install jsmin
```
