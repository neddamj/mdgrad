from mdgrad.tensor import Tensor
from .layers import Module
import numpy as np

class ReLU(Module):
    def forward(self, x):
        assert isinstance(x, Tensor), 'input must be a Tensor'
        x.requires_grad = True
        return x.relu()
    
    def parameters(self):
        return []
    
class Sigmoid(Module):
    def forward(self, x):
        assert isinstance(x, Tensor), 'input must be a Tensor'
        x.requires_grad = True
        return x.sigmoid()
    
    def parameters(self):
        return []
    
class Tanh(Module):
    def forward(self, x):
        assert isinstance(x, Tensor), 'input must be a Tensor'
        x.requires_grad = True
        return x.tanh()
    
    def parameters(self):
        return []

class SoftMax(Module):
    def forward(self, x):
        assert isinstance(x, Tensor), 'input must be a Tensor'
        x.requires_grad = True
        return x.softmax()
    
    def parameters(self):
        return []