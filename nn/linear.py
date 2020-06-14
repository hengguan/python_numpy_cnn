import numpy as np

class Linear:

    def __init__(self, in_channels, out_channels):
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.weights = np.random.randn(in_channels, out_channels)
        self.bias = np.zeros((1.0, out_channels), dtype=np.float)
        self.feature = None
        # self.x_grad = None
        self.dw = None
        self.db = None
    
    def backword(self, dy):
        dx = dy.dot(self.weights.T)
        w_grad = self.feature.T
        self.dw = w_grad.dot(dy)
        self.db = np.mean(dy, axis=0)
        return dx

    def __call__(self, x):
        '''
        b, n = x_shape
        b, m = out_shape
        '''
        self.feature = x.copy()
        out = x.dot(self.weights) + self.bias
        return out