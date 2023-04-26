import torch
from unet import UNet, ConvPass


# Torch model
class LsdModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            output_shapes,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up):

        super().__init__()

        num_fmaps = sum(output_shapes)
        
        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=True,
                num_heads=1)

        self.lsd_head = ConvPass(num_fmaps, output_shapes[0], [[1, 1]], activation='Sigmoid')

    def forward(self, input_raw):

        z = self.unet(input_raw)
        lsds = self.lsd_head(z)

        return lsds


# Torch loss
class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

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
            lsds_weights):

        loss = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)

        return loss


# Make instance of model
model = LsdModel(
        in_channels=1,
        output_shapes=[6],
        fmap_inc_factor=5,
        downsample_factors=[[2,2],[2,2],[2,2]],
        kernel_size_down=[
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]]],
        kernel_size_up=[
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]],
            [[3, 3], [3, 3]]],
        )

# Gunpowder array keys for prediction
inference_array_keys = [
        {
            "RAW": 1
            },
        {
            "PRED_LSDS": 6
            }
        ]

# Gunpowder array keys for training
training_array_keys = [
        {
            "RAW": 1
            },
        {
            "LABELS": 1,
            "UNLABELLED": 1,
            "PRED_LSDS": 6,
            "GT_LSDS": 6,
            "LSDS_WEIGHTS": 6,
            }
        ]

# model input and output shapes in voxels
input_shape = [196, 196]
output_shape = [104, 104]

# default voxel resolution (nm/px)
voxel_size = [8,8]
