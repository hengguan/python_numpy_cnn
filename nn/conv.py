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
            (self.k_size, self.k_size, self.in_channels, self.out_channels))
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

        # print('padding')
        if self.padding:
            out_h, out_w = h, w
            # out = np.zeros((n, h, w, self.out_channels))
            padding = (h - 1)*self.stride - h + self.k_size 
            padding = (int(padding//2), int(padding//2)) if padding % 2 ==0 \
                else (int(padding/2+0.5), int(padding/2-0.5))
            x = np.pad(x, ((0, 0), padding, padding, (0, 0)), mode='constant', constant_values=0)
            # print(x.shape)
        else:
            out_h = int((h-self.k_size)/self.stride + 1)
            out_w = int((h-self.k_size)/self.stride + 1)
            # out_size = (x_shape[1]-self.k_size)/self.stride + 1
            # out = np.zeros((n, int((h-self.k_size)/self.stride + 1), \
            #     int((h-self.k_size)/self.stride + 1), self.out_channels))

        # new_h = x.shape[1]
        # print('convolution')
        out = np.zeros((n, out_h, out_w, self.out_channels))
        # blocks = []# x[:, :self.k_size, :self.k_size, :].reshape(n, 1, -1)
        for i in range(out_h):
            for j in range(out_w):
                sh, sw = i*self.stride, j*self.stride
                block = x[:, sh:sh+self.k_size, sw:sw+self.k_size, :].reshape(n, -1)
                out[:, i, j, :] = block.dot(self.weights).reshape(n, 1, 1, self.out_channels)
        #         blocks.append(block)
        # feature = np.concatenate(blocks, axis=1)
        # out = feature.dot(self.weights).reshape(n, out_h, out_w, self.out_channels)
        # out[:, j, i, :] = row_res.reshape(n, 1, 1, self.out_channels)

        self.x_grad = self.weights
        self.w_grad = self.feature

        return out
