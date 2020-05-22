import numpy as np

class Linear:

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = np.random.randn(in_channels+1, out_channels)
        self.feature = None
        self.x_grad = None
        self.weights_grad = None

    def __call__(self, x):
        '''
        b, n = x_shape
        b, m = out_shape
        '''
        self.feature = x.copy()
        bias = np.ones((x.shape[0], 1))
        x = np.concatenate((x, bias), axis=1)
        out = x.dot(self.weights)

        self.x_grad = self.weights.T
        self.weights_grad = x.T
        return out