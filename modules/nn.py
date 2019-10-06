import numpy as np
import types
import cv2
import time


class Conv2d:
    def __init__(
        self,
        in_channels,
        out_channels, 
        k_size=None, 
        stride=1, 
        padding=True):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_size = k_size
        self.stride = stride
        self.padding = padding

        assert in_channels is not None and out_channels is not None

        self.weights = np.random.rand(self.k_size, self.k_size, self.in_channels, self.out_channels)
        self.bias = 0.0
        self.weights = self.weights.reshape(-1, self.out_channels)

        self.feature = None

    def forward(self, x):
        if not isinstance(x, np.ndarray):
            raise 'input is not numpy.ndarray'
        self.feature = x
        x_shape = x.shape
        assert len(x_shape) == 4
        if len(self.weights) == 0:
            for _ in range(self.out_channels):
                self.w = 0.1*np.random.rand(x_shape[0], self.k_size, self.k_size, self.in_channels)
                self.b = 0.0
                self.weights.append((self.w, self.b))

        print('padding')
        if self.padding:
            out = np.zeros((x_shape[0], x_shape[1], x_shape[2], self.out_channels))
            padding = (x_shape[1] - 1)*self.stride - x_shape[1] + self.k_size 
            padding = (int(padding//2), int(padding//2)) if padding % 2 ==0 else (int(padding/2+0.5), int(padding/2-0.5))
            x = np.pad(x, ((0, 0), padding, padding, (0, 0)), mode='constant', constant_values=0)
            print(x.shape)
        else:
            out_size = (x_shape[0]-self.k_size)/self.stride + 1
            out = np.zeros((x_shape[0], out_size, out_size, self.out_channels))
        print('convolution')
        for i in range(0, x.shape[1]-self.k_size+1, self.stride):
            for j in range(0, x.shape[1]-self.k_size+1, self.stride):
                a = x[:, j:j+self.k_size, i:i+self.k_size, :].reshape(x_shape[0], -1)
                # print(a.shape)
                res = a.dot(self.weights)
                # res = np.array([np.sum(np.sum(np.sum(x[:, j:j+self.k_size, i:i+self.k_size, :]*self.weights[idx][0], axis=1), axis=1), axis=1) for idx in range(self.out_channels)])
                # res = np.reshape(res.T, (x.shape[0], 1, 1, self.out_channels))
                out[:, j, i, :] = res.reshape(x_shape[0], 1, 1, self.out_channels)

        self.x_grad = self.weights
        self.w_grad = self.feature

        return out
        

img = cv2.imread('data/02_002.png')
img = img[490:590, 960:1060, :]
print(img.shape)
img = img / 255.0
x = np.random.rand(32, 32, 3)
layer = Conv2d(3, 64, k_size=3, stride=1, padding=True)
t1 = time.time()
out = layer.forward(np.array([img]))
print('cost time: {}'.format(time.time()-t1))
for i in range(64):
    im = out[0, :, :, i].reshape(100, 100, 1)
    print(im.shape)
    cv2.imshow('img', im)
    cv2.waitKey(500)
