import nn
import numpy as np


class LeNetX(object):

    def __init__(self):
        # build model
        self.conv1 = nn.Conv2d(3, 8, k_size=5, stride=1, padding=0)
        self.relu1 = nn.Relu()
        self.pool1 = nn.MaxPooling(k_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, k_size=3, stride=1, padding=0)
        self.relu2 = nn.Relu()
        self.pool2 = nn.MaxPooling(k_size=2, stride=2)

        self.linear1 = nn.Linear(36, 128)
        self.relu3 = nn.Relu()
        self.linear2 = nn.Linear(128, 84)
        self.relu4 = nn.Relu()
        self.out = nn.Linear(84, 10)
    
    def __call__(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        x = nn.view(x)
        x = self.relu3(self.linear1(x))
        x = self.relu4(self.linear2(x))
        return self.out(x)

    def backward(self, grad):
        pass
        