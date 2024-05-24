from madnet.tensor import Tensor
from .layers import Module
import numpy as np

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert isinstance(x, Tensor), 'input must be a Tensor'
        return x.relu()
    
class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert isinstance(x, Tensor), 'input must be a Tensor'
        return x.sigmoid()

class SoftMax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert isinstance(x, Tensor), 'input must be a Tensor'
        return x.softmax()