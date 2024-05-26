from mdgrad.tensor import Tensor
import numpy as np

class Module:
    def __call__(self, *x):
        out = self.forward(*x)
        return out
    
    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        params = []
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                params.append(value)
            if isinstance(value, Module):
                params.extend(value.parameters())
        return list(set(params))
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.bias = bias
        self.w = Tensor.randn(in_features, out_features) / np.sqrt(in_features + out_features)
        self.b = Tensor.zeros((1, out_features)) if self.bias else None

    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        
        out = x @ self.w + self.b if self.bias else x @ self.w
        # Add previous tensors to computation graph
        out._prev = set((x,))

        def _backward():
            # Gradients of parameters
            self.w.grad += x.transpose() @ out.grad
            if self.bias:
                self.b.grad += out.grad.sum(0, keepdims=True)
            # Gradients of inputs
            x.grad += out.grad @ self.w.transpose()
        out._backward = _backward

        return out

    def parameters(self):
        if self.bias:
            return [self.w, self.b]
        else:
            return [self.w]

class Sequential(Module):
    def __init__(self, layers=[]):
        super().__init__()
        assert isinstance(layers, (list, tuple)), 'input must be a list or tuple'
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params