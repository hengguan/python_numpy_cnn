import numpy as np

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

    def __call__(self, x):
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
