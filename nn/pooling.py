import numpy as np

class MaxPooling:

    def __init__(self, k_size=2, stride=2):
        self.k_size = k_size
        self.stride = stride 

    def __call__(self, x):
        '''
        b, h, w, c = x_shape
        b, (h-k_size)/stride+1, (w-k_size)/stride+1, c = out_shape
        '''
        assert len(x.shape)==4, "shape format of input is not [n, h, w, c]"
        n, h, w, c = x.shape
        out_h = (h-self.k_size)/self.stride+1
        out_w = (w-self.k_size)/self.stride+1

        if int(out_h)!=out_h or int(out_w)!=out_w:
            out_h = np.floor(out_h)
            out_w = np.floor(out_w)
            new_h = (out_h-1)*self.stride+self.k_size
            new_w = (out_w-1)*self.stride+self.k_size
            x = np.pad(x, ((), (new_h-h,), (new_w-w,), ()), 'edge')
     
        out = np.zeros((n, int(out_h), int(out_w), c))
        for i in range(0, int(out_h)):
            for j in range(0, int(out_w)):
                hs, ws = i*self.stride, j*self.stride
                max_x = np.max(
                    x[:, hs:hs+self.k_size, ws:ws+self.k_size, :].reshape(n, -1, c), 
                    axis=1)
                out[:, i, j, :] = max_x.reshape(n, 1, 1, c)
        
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
