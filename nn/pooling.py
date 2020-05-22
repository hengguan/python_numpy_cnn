import numpy as np

class MaxPooling:

    def __init__(self, k_size=None, stride=None):
        self.k_size = k_size
        self.stride = stride 

    def __call__(self, x):
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

    def __call__(self, x):
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
