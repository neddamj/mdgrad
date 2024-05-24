from madnet.tensor import Tensor
from .layers import Module
import numpy as np

class SGD(Module):
    def __init__(self, parameters, lr=0.001):
        super().__init__()
        self.params = parameters
        self.lr = lr

    def __call__(self):
        self.step()

    def parameters(self):
        return self.params
        
    def zero_grad(self):
        return super().zero_grad()
    
    def step(self):
        for param in self.parameters():
            param.data = param.data - self.lr * param.grad