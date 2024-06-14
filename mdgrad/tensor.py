import numpy as np
import math

class Tensor:
    def __init__(self, data, _children=(), requires_grad=False):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = None #np.zeros_like(self.data)
        # Set data to floats to division can be done
        self.data = self.data.astype(np.float32) 
        #self.grad = self.grad.astype(np.float32)
        self.requires_grad = requires_grad
        self._prev = set(_children)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self._backward = lambda: None
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=True)
        out = Tensor(self.data + other.data, (self, other), requires_grad=True)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):    
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=True)
        out = Tensor(self.data - other.data, (self, other), requires_grad=True)

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = np.ones_like(self.data) * other
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=True)
        out = Tensor(self.data * other.data, (self, other), requires_grad=True)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be a scalar (int/float)"
        out = Tensor(self.data ** other, (self,), requires_grad=True)

        def _backward():
            self.grad += (other * self.data ** (other -1)) * out.grad
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=True)
        out = Tensor(self.data @ other.data, (self, other), requires_grad=True)

        def _backward():
            self.grad += out.grad @ np.transpose(other.data)
            other.grad += np.transpose(self.data) @ out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = self.data * (self.data > 0)
        out = Tensor(self.data * (self.data > 0), (self,), requires_grad=True)

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes=axes), (self,), requires_grad=True)
        
        def _backward():
            self.grad += np.transpose(out.grad, axes=axes)
        out._backward = _backward
        
        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-9), (self,), requires_grad=True)

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        exp = np.exp(self.data)
        out = Tensor(exp, (self,), requires_grad=True)
        
        def _backward():
            self.grad += exp * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        def _positive(x):
            return 1 / (1 + np.exp(-x))
        def _negative(x):
            exp = np.exp(x)
            return exp / (exp + 1)
        # Numerically stable sigmoid
        positive = self.data >= 0
        negative = ~positive
        value = np.empty_like(self.data)
        value[positive] = _positive(self.data[positive])
        value[negative] = _negative(self.data[negative])
        out = Tensor(value, (self,), requires_grad=True)
        
        def _backward():
            #exp = np.exp(-self.data)
            g = value * (1 - value)
            self.grad += g * out.grad
        out._backward = _backward

        return out
    
    def tanh(self, dim=1):
        val = np.tanh(self.data)
        out = Tensor(val, (self,), requires_grad=True)

        def _backward():
            self.grad += (1 - val**2) * out.grad
        out._backward = _backward

        return out
    
    def softmax(self, dim=1):
        exps = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        probs = exps / np.sum(exps, axis=dim, keepdims=True)
        out = Tensor(probs, (self,), requires_grad=True)

        def _backward():
            for i, (output, grad) in enumerate(zip(out.data, out.grad)):
                output = output.reshape(-1, 1)
                jacobian = np.diagflat(output) - output @ np.transpose(output)
                g = np.transpose(jacobian @ np.transpose(grad))
                self.grad[i] += g
        out._backward = _backward

        return out
    
    def reshape(self, *new_shape):
        old_shape = self.shape
        out = Tensor(self.data.reshape(*new_shape), (self,), requires_grad=True)

        def _backward():
            self.grad += out.grad.reshape(old_shape)
        out._backward = _backward

        return out
    
    def sum(self, axis=None, keepdims=np._NoValue):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), requires_grad=True)

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward

        return out
    
    def abs(self):
        out = Tensor(np.abs(self.data), (self,), requires_grad=True)

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward

        return out
    
    def sin(self):
        out = Tensor(np.sin(self.data), (self,), requires_grad=True)

        def _backward():
            self.grad += np.cos(self.data) * out.grad
        out._backward = _backward

        return out
    
    def cos(self):
        out = Tensor(np.cos(self.data), (self,), requires_grad=True)

        def _backward():
            self.grad += -np.sin(self.data) * out.grad
        out._backward = _backward

        return out
        
    def backward(self):
        # https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited and v.requires_grad:
                visited.add(v)
                if v.grad is None:
                    v.grad = np.zeros_like(v.data)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        if self.grad is None:
            if self.size != 1:
                raise RuntimeError('grad can be implicitly created only for scalar outputs')
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward() 
    
    def max(self, axis=None):
        return np.max(self.data, axis=axis)
    
    def min(self, axis=None):
        return np.min(self.data, axis=axis)
    
    def numpy(self):
        return self.data.copy()
    
    def tolist(self):
        return self.data.copy().tolist()
    
    def item(self):
        assert self.data.size == 1, f'The data has {self.data.size} elements when it should only have 1'
        return self.data.copy().item()
        
    @classmethod
    def zeros(cls, shape, requires_grad=False):
        assert isinstance(shape, int) or isinstance(shape, tuple), f'shape should be int or tuple insted of {type(shape)}'
        return cls(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape, requires_grad=False):
        assert isinstance(shape, int) or isinstance(shape, tuple), f'shape should be int or tuple insted of {type(shape)}'
        return cls(np.ones(shape), requires_grad=requires_grad)
    
    @classmethod
    def normal(cls, mean=0.0, std=1.0, shape=None, requires_grad=False):
        assert isinstance(shape, int) or isinstance(shape, tuple), f'shape should be int or tuple insted of {type(shape)}'
        return cls(np.random.normal(mean, std, shape), requires_grad=requires_grad)
    
    @classmethod
    def randn(cls, *args, requires_grad=False):
        return cls(np.random.randn(*args), requires_grad=requires_grad)
    
    @classmethod
    def eye(cls, N, M=None):
        return cls(np.eye(N, M))

    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other - self
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __repr__(self):
        return f'Tensor({self.data}, dtype={self.data.dtype})'
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        # Track the current element in the iterable
        self.current = 0
        return self
    
    def __next__(self):
        if self.current >= len(self.data):
            raise StopIteration
        current = self.data[self.current]
        self.current += 1
        return current
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

def sum(x, axis=None, keepdims=np._NoValue):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return x.sum(axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=np._NoValue):
    x = x if isinstance(x, Tensor) else Tensor(x)
    out = x.sum(axis=axis, keepdims=keepdims) 
    return out * math.prod(out.shape) / math.prod(x.shape)

def var(x, axis=None, keepdims=np._NoValue):
    x = x if isinstance(x, Tensor) else Tensor(x)
    mean_val = mean(x, axis=axis, keepdims=keepdims)
    out = mean((x - mean_val)**2, axis=axis, keepdims=keepdims)
    return out

def std(x, axis=None, keepdims=np._NoValue):
    x = x if isinstance(x, Tensor) else Tensor(x)
    out = var(x, axis=axis, keepdims=keepdims) ** 0.5
    return out

def exp(x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return x.exp()

def log(x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return x.log()

def sin(x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return x.sin()

def cos(x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return x.cos()

def max(x, axis=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return x.max(axis=axis)

def min(x, axis=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return x.min(axis=axis)

def argmax(x, axis=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return Tensor(np.argmax(x.data, axis=axis))

def argmin(x, axis=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return Tensor(np.argmin(x.data, axis=axis))

def transpose(x, axes=None):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return x.transpose(axes=axes)

def ones(shape, requires_grad=False):
    return Tensor.ones(shape=shape, requires_grad=requires_grad)

def zeros(shape, requires_grad=False):
    return Tensor.zeros(shape=shape, requires_grad=requires_grad)

def eye(N, M=None):
    return Tensor.eye(N=N, M=M)

def randn(*shape, requires_grad=False):
    return Tensor.randn(*shape, requires_grad=requires_grad)

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)