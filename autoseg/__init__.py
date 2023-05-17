from .train import train
from .predict import predict
from autoseg.segment.hierarchical import run as run_hierarchical
from . import utils

from .data.make_unlabelled_mask import make_mask

import os

models_dir = os.path.join(
    os.path.dirname(__file__), 
    "models"
)

model_paths = {}
for model_group in os.listdir(models_dir):
    model_paths[model_group] = {}
    for model in os.listdir(os.path.join(models_dir, model_group)):
        model_paths[model_group][model] = os.path.join(
            models_dir,
            model_group,
            model,
            "model.py"
        )
