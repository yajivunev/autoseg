from .train import train
from .predict import predict
from . import segment
from . import utils

from .data.make_unlabelled_mask import make_mask

import os

models_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "models"
))

print("list models: ", os.listdir(models_dir))
print("membrane models: ", os.listdir(os.path.join(models_dir,"membrane")))

model_paths = {}
for model_group in [x for x in os.listdir(models_dir) if ('__init__' not in x and '__pycache__' not in x)]:
    print("model group: ",model_group)
    model_paths[model_group] = {}
    for model in [x for x in os.listdir(os.path.join(models_dir, model_group)) if ('__init__' not in x and '__pycache__' not in x)]:
        print("model name: ", model)
        model_paths[model_group][model] = os.path.join(
            models_dir,
            model_group,
            model,
            "model.py"
        )
