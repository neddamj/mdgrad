from mdgrad.tensor import Tensor
from .layers import Module
import numpy as np

class MSELoss(Module):
    def forward(self, x, y):
        assert x.shape == y.shape, 'input and target tensors must be the same shape'
        x = x if isinstance(x, Tensor) else Tensor(x)
        y = y if isinstance(y, Tensor) else Tensor(y)
        
        out = ((x - y) ** 2).sum() /  x.size

        return out
    
    def parameters(self):
        return []
    
class CrossEntropyLoss(Module):
    def forward(self, x, y):
        assert x.shape == y.shape, 'input and target tensors must be the same shape'
        x = x if isinstance(x, Tensor) else Tensor(x)
        y = y if isinstance(y, Tensor) else Tensor(y)
        probs = x.softmax()
        

    def backward(self):
        pass