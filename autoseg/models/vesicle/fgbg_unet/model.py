import json
import torch
from funlib.learn.torch.models import UNet, ConvPass


torch.backends.cudnn.benchmark = True


# 2D UNet
class Model(torch.nn.Module):

    def __init__(
            self,
            config_path="config.json"):

        super().__init__()

        self.config_path = config_path
        self.load_config()

        self.unet = UNet(**self.params)
        self.mask_head = ConvPass(self.unet.out_channels, self.output_shapes[0], [[1, 1]], activation='Sigmoid')
        
        self.check_output_shape()

    def load_config(self):

        with open(self.config_path,"r") as f: 
            config = json.load(f)

        for k,v in config["model"].items():
            if type(v) == str:
                value = f'"{v}"'
            else:
                value = str(v)

            exec(f'self.{k} = {value}')

    def check_output_shape(self):

        input_shape = [1,1] + self.input_shape

        try:
            out_shape = self.output_shape
        except:
            print("Computing output shape for first time for input_shape...")
            self.output_shape = self.forward(torch.rand(*(input_shape))).data.shape[2:]

            with open(self.config_path,"r+") as f: 
                config = json.load(f)

                config["model"]["output_shape"]  = self.output_shape

    def return_optimizer(self):
        return getattr(torch.optim,self.optimizer["name"])

    def return_loss(self):
        return Loss()

    def forward(self, input_raw):

        z = self.unet(input_raw)

        mask = self.mask_head(z)

        return mask


# Weighted MSE Loss
class Loss(torch.nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss

    def forward(
            self,
            prediction,
            target,
            weights):

        loss = self._calc_loss(prediction, target, weights)

        return loss
