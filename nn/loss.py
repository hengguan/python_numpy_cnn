import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        self.feature = None
    
    def __call__(self, x, target):
        self.feature = x.copy()
        out = np.sum(-target*np.log(x))
        return out