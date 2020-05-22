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
            out = np.zeros((n, h, w, self.out_channels))
            padding = (h - 1)*self.stride - h + self.k_size 
            padding = (int(padding//2), int(padding//2)) if padding % 2 ==0 \
                else (int(padding/2+0.5), int(padding/2-0.5))
            x = np.pad(x, ((0, 0), padding, padding, (0, 0)), mode='constant', constant_values=0)
            print(x.shape)
        else:
            # out_size = (x_shape[1]-self.k_size)/self.stride + 1
            out = np.zeros((n, int((h-self.k_size)/self.stride + 1), \
                int((h-self.k_size)/self.stride + 1), self.out_channels))

        new_h = x.shape[1]
        print('convolution')
        for i in range(0, new_h-self.k_size+1, self.stride):
            for j in range(0, new_h-self.k_size+1, self.stride):
                row = x[:, j:j+self.k_size, i:i+self.k_size, :].reshape(n, -1)
                row_res = row.dot(self.weights)
                out[:, j, i, :] = row_res.reshape(n, 1, 1, self.out_channels)
                
        self.x_grad = self.weights
        self.w_grad = self.feature

        return out
