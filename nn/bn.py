import numpy as np

class BatchNorm:
    """
    mean
    """
    def __init__(self):
        # self.weights = np.random.random()
        # self.bias = np.random.random()
        self.epislon = 10e-5
        self.mean_grad = None
        self.sigma_power_grad = None
        self.x_hat_grad = None

        self.gamma = None
        self.beta = None
        self.gamma_grad = None
        self.beta_grad = None

    def __call__(self, x):
        '''
        b, h, w, c = x.shape
        '''
        assert len(x.shape)==4, "shape format of input is not [n, h, w, c]"
        n, h, w, c = x.shape
        assert n>1, "batch size must be more than 1 when using batch normalization"
        # batch_size = x.shape[0]
        if self.gamma is None:
            self.gamma = np.ones((n, 1))
            self.beta = np.zeros((n, 1))
        feature = x.reshape(n, -1)
        mean = np.mean(feature, axis=-1, keepdims=True)

        sigma_power = np.mean((feature-mean)**2, axis=-1, keepdims=True)
        x_hat = np.divide((feature-mean), np.sqrt(sigma_power+self.epislon))
        y = self.gamma*x_hat+self.beta

        self.gamma_grad = np.sum(x_hat, axis=-1, keepdims=True)
        self.beta_grad = np.ones_like(self.beta)
        self.x_hat_grad = self.gamma*np.ones_like(x_hat)

        return y.reshape(n, h, w, c)

if __name__ == "__main__":
    import time
    x = np.random.randn(2, 20, 20, 3)
    print(x)
    bn = BatchNorm()
    ts = time.time()
    y = bn(x)
    print("cost time: {}".format(time.time()-ts))
    print(y)
    # print(np.mean(y))