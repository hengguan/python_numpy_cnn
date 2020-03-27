import numpy as np
import types
import cv2
import time
import matplotlib.pyplot as plt


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

        self.weights = 0.1*np.random.standard_normal((self.k_size, self.k_size, self.in_channels, self.out_channels))
        self.bias = 0.1 * np.random.standard_normal(self.out_channels)
        self.weights = self.weights.reshape(-1, self.out_channels)
        print(self.weights)
        self.feature = None

    def forward(self, x):
        if not isinstance(x, np.ndarray):
            raise 'input is not numpy.ndarray'
        self.feature = x
        x_shape = x.shape
        assert len(x_shape) == 4

        # print('padding')
        if self.padding:
            out = np.zeros((x_shape[0], x_shape[1], x_shape[2], self.out_channels))
            padding = (x_shape[1] - 1)*self.stride - x_shape[1] + self.k_size 
            padding = (int(padding//2), int(padding//2)) if padding % 2 ==0 else (int(padding/2+0.5), int(padding/2-0.5))
            x = np.pad(x, ((0, 0), padding, padding, (0, 0)), mode='constant', constant_values=0)
            print(x.shape)
        else:
            # out_size = (x_shape[1]-self.k_size)/self.stride + 1
            out = np.zeros((x_shape[0], int((x_shape[1]-self.k_size)/self.stride + 1), int((x_shape[1]-self.k_size)/self.stride + 1), self.out_channels))

        print('convolution')
        for i in range(0, x.shape[1]-self.k_size+1, self.stride):
            for j in range(0, x.shape[1]-self.k_size+1, self.stride):
                row = x[:, j:j+self.k_size, i:i+self.k_size, :].reshape(x_shape[0], -1)
                row_res = row.dot(self.weights)
                out[:, j, i, :] = row_res.reshape(x_shape[0], 1, 1, self.out_channels)
                
        self.x_grad = self.weights
        self.w_grad = self.feature

        return out


class BatchNorm:
    def __init__(self):
        self.weights = np.random.random()
        self.bias = np.random.random()
        self.epislon = 0.0000001
        self.mean_grad = None
        self.sigma_power_grad = None
        self.x_hat_grad = None

        self.weights_grad = None
        self.bias_grad = None

    def forward(self, x):
        '''
        b, h, w, c = x.shape
        '''
        batch_size = x.shape[0]
        if self.weights is None:
            self.weights = 0.1*np.random.standard_normal((1, batch_size))
            self.bias = np.random.rand(1, batch_size)
        mean = np.mean(x)
        sigma_power = sum([(xi-mean)**2 for xi in x])/batch_size
        x_hat = np.array([(xi-mean)/np.sqrt(sigma_power+self.epislon) for xi in x])
        y = self.weights*x_hat+self.bias

        self.weights_grad = x_hat
        self.bias_grad = 1.0
        self.x_hat_grad = self.weights
        self.sigma_power_grad = np.array([-0.5*(xi-mean)*((sigma_power+self.epislon)**(-1.5)) for xi in x])
        self.mean_grad = np.array([-(sigma_power+self.epislon)**(-0.5) for xi in x])

        return y


class Relu:
    def __init__(self, inplace=True):
        self.in_place = inplace
        self.feature = None
        self.x_grad = None

    def forward(self, x):
        self.feature = x.copy()
        x[x<0] = 0
        out = x.copy()
        x[x>0] = 1
        self.x_grad = x
        return out 


class MaxPooling:
    def __init__(self, k_size=None, stride=None):
        self.k_size = k_size
        self.stride = stride 

    def forward(self, x):
        '''
        b, h, w, c = x_shape
        b, (h-k_size)/stride+1, (w-k_size)/stride+1, c = out_shape
        '''
        h, w = (x.shape[1]-self.k_size)/self.stride+1, (x.shape[2]-self.k_size)/self.stride+1
        out = np.zeros((x.shape[0], h, w, x.shape[-1]))
        for i in range(0, x.shape[2]-self.k_size+1, self.stride):
            for j in range(0, x.shape[1]-self.k_size+1, self.stride):
                max_x = np.max(x[:, j:j+self.k_size, i:i+self.k_size, :].reshape(x.shape[0], -1, x.shape[-1]), axis=1)
                out[:, j:j+self.k_size, i:i+self.k_size, :] = max_x.reshape(x.shape[0], self.k_size, self.k_size, x.shape[-1])
        
        return out


class AvePooling:
    def __init__(self, k_size=None, stride=None):
        self.k_size = k_size
        self.stride = stride 

    def forward(self, x):
        '''
        b, h, w, c = x_shape
        b, (h-k_size)/stride+1, (w-k_size)/stride+1, c = out_shape
        '''
        h, w = (x.shape[1]-self.k_size)/self.stride+1, (x.shape[2]-self.k_size)/self.stride+1
        out = np.zeros((x.shape[0], h, w, x.shape[-1]))
        for i in range(0, x.shape[2]-self.k_size+1, self.stride):
            for j in range(0, x.shape[1]-self.k_size+1, self.stride):
                mean_x = np.mean(x[:, j:j+self.k_size, i:i+self.k_size, :].reshape(x.shape[0], -1, x.shape[-1]), axis=1)
                out[:, j:j+self.k_size, i:i+self.k_size, :] = mean_x.reshape(x.shape[0], self.k_size, self.k_size, x.shape[-1])
        
        return out


class Linear:
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = np.random.randn(in_channels+1, out_channels)
        self.feature = None
        self.x_grad = None
        self.weights_grad = None

    def forward(self, x):
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


class Softmax:
    def __init__(self):
        self.feature = None

    def forward(self, x):
        self.feature = x.copy()
        if len(x.shape)!= 2:
            x = x.reshape(x.shape[0], -1)
        exp_x = np.exp(x)
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        softmax_x = exp_x/sum_exp
        return softmax_x


class CrossEntropyLoss:
    def __init__(self):
        self.feature = None
    
    def forward(self, x, target):
        self.feature = x.copy()
        out = np.sum(-target*np.log(x))
        return out

        
if __name__ == "__main__":
    print('shit change git test')
    img = cv2.imread('data/cats.jpg')
    img = img[:900, 250:1150, :]
    print(img.shape)
    img = img / 255.0
    plt.imshow(img)
    plt.show()
    x = np.random.rand(32, 32, 3)
    layer = Conv2d(3, 64, k_size=3, stride=1, padding=0)
    relu = Relu()
    t1 = time.time()
    out = layer.forward(np.array([img]))
    t2 = time.time()
    print('cost time: {}'.format(t2-t1))
    out = relu.forward(out)
    print('relu cost: ', time.time()-t2)
    for i in range(64):
        im = out[0, :, :, i].reshape(out.shape[1], out.shape[2], 1)
        print(im.shape)
        cv2.imshow('img', im)
        cv2.waitKey(500)
