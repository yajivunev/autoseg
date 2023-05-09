import gunpowder as gp
import numpy as np


class CreateMask(gp.BatchFilter):
    def __init__(self, labels, gt_mask):
        self.labels = labels
        self.gt_mask = gt_mask

    def setup(self):
        # tell downstream nodes about the new array
        self.provides(self.gt_mask, self.spec[self.labels].copy())

    def prepare(self, request):
        # to deliver inverted raw data, we need raw data in the same ROI
        deps = gp.BatchRequest()
        deps[self.labels] = request[self.gt_mask].copy()

        return deps

    def process(self, batch, request):
        labels = batch[self.labels].data

        gt_mask = (labels > 0).astype(np.float32)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.gt_mask].roi.copy()
        spec.dtype = np.float32

        batch = gp.Batch()

        batch[self.gt_mask] = gp.Array(gt_mask, spec)

        return batch
