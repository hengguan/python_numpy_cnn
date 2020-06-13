import numpy as np

from .activatefn import Relu, Softmax
from .bn import BatchNorm
from .conv import Conv2d
from .linear import Linear
from .pooling import MaxPooling, AvePooling
from .loss import CrossEntropyLoss

def view(x):
    if not isinstance(x, np.ndarray):
            raise ValueError(
        'dtype of input is not numpy.ndarray')
    assert len(x.shape)==4, \
        "the shape of input must be format as [n, h, w, c]"
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)
    