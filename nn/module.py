import numpy as np

class Module(object):

    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError(
    "The __call__ method of every child subclass must be Implemented")
        