import numpy as np

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

        self.weights = 0.1*np.random.standard_normal(
            (self.in_channels, self.k_size, self.k_size, self.out_channels))
        self.bias = 0.1 * np.random.standard_normal(self.out_channels)
        self.weights = self.weights.reshape(-1, self.out_channels)
        
        self.feature = None

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            raise ValueError('dtype of input is not numpy.ndarray')
        assert len(x.shape)==4, "the shape of input must be format as [n, h, w, c]"
        self.feature = x
        n, h, w, c = x.shape
        assert c==self.in_channels, \
            "the channel({}) of input feature is equal to the channel({}) of weights." \
                .format(c, self.in_channels)
        print(self.feature.shape, x.shape)
        # print('padding')
        if self.padding:
            temp_h, temp_w = h/self.stride, w/self.stride
            out_h = int(temp_h)+1 if (temp_h)%1!=0 else int(temp_h)
            out_w = int(temp_w)+1 if (temp_w)%1!=0 else int(temp_w)
            # out = np.zeros((n, h, w, self.out_channels))
            pad_h = (out_h - 1)*self.stride - h + self.k_size 
            pad_w = (out_w - 1)*self.stride - w + self.k_size
            self.padding_h = int(pad_h//2) if pad_h % 2 ==0 else int(pad_h/2)+1
            self.padding_w = int(pad_w//2) if pad_w % 2 ==0 else pad_w-self.padding_h
            pad = (self.padding_h, self.padding_w)
            x = np.pad(x, ((0, 0), pad, pad, (0, 0)), mode='constant', constant_values=0)
            # print(x.shape)
        else:
            out_h = int((h-self.k_size)/self.stride + 1)
            out_w = int((w-self.k_size)/self.stride + 1)
            self.padding_h, self.padding_w = 0, 0
            # out_size = (x_shape[1]-self.k_size)/self.stride + 1
            # out = np.zeros((n, int((h-self.k_size)/self.stride + 1), \
            #     int((h-self.k_size)/self.stride + 1), self.out_channels))
        print(self.feature.shape, x.shape)
        # new_h = x.shape[1]
        # print('convolution')
        out = np.zeros((n, out_h, out_w, self.out_channels))
        # blocks = []# x[:, :self.k_size, :self.k_size, :].reshape(n, 1, -1)
        for i in range(out_h):
            sh = i*self.stride
            for j in range(out_w):
                sh = j*self.stride
                block = x[:, sh:sh+self.k_size, sw:sw+self.k_size, :].reshape(n, -1)
                out[:, i, j, :] = block.dot(self.weights).reshape(n, 1, 1, self.out_channels)
        #         blocks.append(block)
        # feature = np.concatenate(blocks, axis=1)
        # out = feature.dot(self.weights).reshape(n, out_h, out_w, self.out_channels)
        # out[:, j, i, :] = row_res.reshape(n, 1, 1, self.out_channels)

        # self.x_grad = self.weights
        self.w_grad = x

        return out

        def backword(self, dy):
            yn, yh, yw, yc = dy.shape
            out_c = self.w_grad.shape[-1]
            x = np.zeros_like(self.w_grad)
            self.w_grad = np.zeros_like(self.weights)
            for i in yh:
                xh = i*self.stride
                for j in yw:
                    xw = j*self.stride
                    grad_y = dy[:, i, j, :].reshape(yn, 1, yc)
                    grad_x = (grad_y * self.weights).reshape(yn, self.k_size, self.k_size, out_c)
                    x[:, xh:xh+self.k_size, xw:xw+self.k_size, :] += grad_x

