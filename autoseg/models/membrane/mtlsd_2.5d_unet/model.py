import json
import torch
from unet import UNet, ConvPass


torch.backends.cudnn.benchmark = True


# 3D UNet
class Model(torch.nn.Module):

    def __init__(
            self,
            config_path="config.json"):

        super().__init__()

        self.config_path = config_path
        self.load_config()

        self.unet = UNet(**self.params)

        self.lsd_head = ConvPass(self.unet.out_channels, self.output_shapes[0], [[1, 1, 1]], activation='Sigmoid')
        self.aff_head = ConvPass(self.unet.out_channels, self.output_shapes[1], [[1, 1, 1]], activation='Sigmoid')

    def load_config(self):

        with open(self.config_path,"r") as f: 
            config = json.load(f)

        for k,v in config["model"].items():
            if type(v) == str:
                value = f'"{v}"'
            else:
                value = str(v)

            exec(f'self.{k} = {value}')

    def return_optimizer(self):
        return getattr(torch.optim,self.optimizer["name"])

    def return_loss(self):
        return Loss()

    def forward(self, input_raw):

        z = self.unet(input_raw)

        lsds = self.lsd_head(z[0])
        affs = self.aff_head(z[1])

        return lsds,affs


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
            lsds_prediction,
            lsds_target,
            lsds_weights,
            affs_prediction,
            affs_target,
            affs_weights):

        lsds_loss = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)
        affs_loss = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return lsds_loss + affs_loss
