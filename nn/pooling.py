import numpy as np

class MaxPooling:

    def __init__(self, k_size=2, stride=2, padding=0):
        self.k_size = k_size
        self.stride = stride 
        self.padding = padding

    def backword(self, dy):
        n, h, w, c = dy.shape
        for i in range(h):
            hs = i*self.stride
            he = hs+self.k_size
            for j in range(w):
                ws = i*self.stride
                we = ws+self.k_size
                self.x_grad[:, hs:he, ws:we, :] *= dy[:, i, j, :].reshape(n, 1, 1, c)
        return self.x_grad[:, :self.height, :self.width, :]


    def __call__(self, x):
        '''
        b, h, w, c = x_shape
        b, (h-k_size)/stride+1, (w-k_size)/stride+1, c = out_shape
        '''
        assert len(x.shape)==4, "shape format of input is not [n, h, w, c]"
        n, h, w, c = x.shape
        self.height, self.width = h, w
        out_h = (h-self.k_size)/self.stride+1
        out_w = (w-self.k_size)/self.stride+1
        int_out_h, int_out_w = int(out_h), int(out_w)
        if self.padding and (int_out_h!=out_h or int_out_w!=out_w):
            int_out_h = int_out_h+1 if int_out_h!=out_h else int_out_h
            int_out_w = int_out_w+1 if int_out_w!=out_w else int_out_w
            pad_h = (int_out_h - 1)*self.stride - h + self.k_size 
            pad_w = (int_out_w - 1)*self.stride - w + self.k_size
            
            x = np.pad(x, ((), (0,pad_h), (0,pad_w), ()), 'edge')
     
        out = np.zeros((n, int_out_h, int_out_w, c))
        self.x_grad = np.zeros_like(x)
        for i in range(int_out_h):
            hs = i*self.stride
            he = hs+self.k_size
            for j in range(int_out_w):
                ws = j*self.stride
                we = ws+self.k_size
                kernel_x = x[:, hs:he, ws:we, :].reshape(n, -1, c)
                # kernel_dx = self.x_grad[:, hs:he, ws:we, :].reshape(n, -1, c)
                max_val = np.max(kernel_x, axis=1)
                out[:, i, j, :] = max_val

                # res = np.argwhere(kernel_x==max_val.reshape(n, 1, c))
                # idx_sort = np.array(sorted(
                #     sorted(res, key=lambda i: i[-1]), 
                #     key=lambda j: j[0]))
                # # print(idx_sort)
                # kernel_dx[idx_sort[:, 0], idx_sort[:, 1], idx_sort[:,2]] = 1.0
                # self.x_grad[:, hs:he, ws:we, :] = kernel_dx.reshape(
                #     n, self.k_size, self.k_size, c)
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
