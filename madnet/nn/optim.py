from madnet.tensor import Tensor
from .layers import Module
import numpy as np

class SGD(Module):
    def __init__(self, parameters, lr=0.001):
        super().__init__()
        self.parameters = parameters
    
    def forward(self):
        raise NotImplementedError