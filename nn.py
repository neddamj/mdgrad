from tensor import Tensor
import numpy as np

#######################################
############# Layers ##################
#######################################
class Module:
    def __init__(self):
        pass

    def __call__(self, *x):
        out = self.forward(*x)
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    
    def forward(self, x):
        pass

    def parameters(self):
        return []
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.bias = bias
        self.w = Tensor.randn(in_features, out_features)
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
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.parameters())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return self.params
 
#######################################
######### Loss Functions ##############
#######################################
class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        assert x.shape == y.shape, 'input and target tensors must be the same shape'
        x = x if isinstance(x, Tensor) else Tensor(x)
        y = y if isinstance(y, Tensor) else Tensor(y)
        
        out = ((x - y) ** 2).sum() /  x.size

        return out
    
#######################################
###### Activation Functions ###########
#######################################
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