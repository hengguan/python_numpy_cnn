import numpy as np
import copy

class Relu:
    
    def __init__(self, inplace=True):
        self.in_place = inplace
        self.feature = None
        self.x_grad = None

    def backword(self, dy):
        self.feature[self.feature>0] = 1
        out = self.feature*dy
        return out

    def __call__(self, x):
        out = np.maximum(x, 0)
        self.feature = out
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

if __name__ == "__main__":
    data = 0.1*np.random.standard_normal(
            (3, 3, 2))

    dy = 0.1*np.random.standard_normal(
            (3, 3, 2))
    relu = Relu()
    print(dy)
    data = relu(data)
    print("----------")
    dout = relu.backword(dy)
    print(dout)