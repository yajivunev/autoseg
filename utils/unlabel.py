import gunpowder as gp
import numpy as np

  
class Unlabel(gp.BatchFilter):
    def __init__(self, labels, unlabelled):
        self.labels = labels
        self.unlabelled = unlabelled

    def setup(self):
        
        spec = self.spec[self.labels].copy()
        self.provides(self.unlabelled, spec) 

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.labels] = request[self.unlabelled].copy()

        return deps

    def process(self, batch, request):
        labels = batch[self.labels].data

        unlabelled = (labels > 0).astype#(np.uint8)

        spec = batch[self.labels].spec.copy()
        spec.roi = request[self.unlabelled].roi.copy()
        #spec.dtype = np.uint8

        batch = gp.Batch()

        batch[self.unlabelled] = gp.Array(unlabelled, spec)

        return batch

