import numpy as np

class Relu:
    
    def __init__(self, inplace=True):
        self.in_place = inplace
        self.feature = None
        self.x_grad = None

    def __call__(self, x):
        self.feature = x.copy()
        x[x<0] = 0
        out = x.copy()
        x[x>0] = 1
        self.x_grad = x
        return out 
        
class Softmax:

    def __init__(self):
        self.feature = None

    def __call__(self, x):
        self.feature = x.copy()
        if len(x.shape)!= 2:
            x = x.reshape(x.shape[0], -1)
        exp_x = np.exp(x)
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        softmax_x = exp_x/sum_exp
        return softmax_x