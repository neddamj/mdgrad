from tensor import Tensor
import numpy as np

class Module:
    def __init__(self):
        pass

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def params(self):
        return []
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.bias = bias
        self.w = Tensor.randn(in_features, out_features)
        self.b = Tensor.zeros((1, out_features)) if self.bias else None

    def __call__(self, x):
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
