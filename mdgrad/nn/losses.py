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
        x = x if isinstance(x, Tensor) else Tensor(x)
        y = y if isinstance(y, Tensor) else Tensor(y)

        # Turn the logits into probs
        probs = x.softmax()
        samples = len(probs)
        probs_clipped = np.clip(probs.numpy(), 1e-7, 1 - 1e-7)
        if len(y.shape) == 1:
            # If y values are class indices
            confidence = probs_clipped[range(samples), y.numpy().astype(int)]
        elif len(y.shape) == 2:
            # If y values are one-hot encoded
            confidence = np.sum(probs_clipped * y.numpy(), axis=1)
        nll = -np.log(confidence)
        out = Tensor(np.mean(nll), (x,))

        def _backward():
            nonlocal y
            if len(y.shape) == 2:
                # Turn labels to discrete values if they are
                # one-hot encoded
                y = np.argmax(y.numpy(), axis=1)
            # Gradients of inputs
            x.grad[np.arange(len(y.data)), y.astype(int)] -= 1
            x.grad += probs / len(y.data) * out.grad

        out._backward = _backward

        return out
        
    def parameters(self):
        return []