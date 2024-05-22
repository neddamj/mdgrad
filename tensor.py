import numpy as np

class Tensor:
    def __init__(self, data, _children=()):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = np.zeros_like(self.data)
        # Set data to floats to division can be done
        self.data = self.data.astype(np.float32) 
        self.grad = self.grad.astype(np.float32)
        self._prev = set(_children)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self._backward = lambda: None
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):    
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be a scalar (int/float)"
        out = Tensor(self.data ** other, (self,))

        def _backward():
            self.grad += (other * self.data ** (other -1)) * out.grad
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ np.transpose(other.data)
            other.grad += np.transpose(self.data) @ out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = self.data * (self.data > 0)
        out = Tensor(self.data * (self.data > 0), (self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def transpose(self,):
        out = Tensor(np.transpose(self.data), (self,))
        
        def _backward():
            self.grad += np.transpose(out.grad)
        out._backward = _backward
        
        return out
    
    def log(self):
        print(self.data)
        val = Tensor(np.log(self.data) + 1e-9, (self,))
        print(val)
        out = val

        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,))
        
        def _backward():
            self.grad += np.exp(self.data) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        value = 1/(1 + np.exp(-self.data))
        out = Tensor(value, (self,))
        
        def _backward():
            exp = np.exp(-self.data)
            g = exp/((1+exp)**2)
            self.grad += g * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        val = (np.exp(self.data) - np.exp(-self.data))/(np.exp(self.data) + np.exp(-self.data))
        out = Tensor(val, (self,))

        def _backward():
            self.grad += (1 - val**2) * out.grad
        out._backward = _backward

        return out
    
    def softmax(self, dim=1):
        exps = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        probs = exps / np.sum(exps, axis=dim, keepdims=True)
        out = Tensor(probs, (self,))

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
        out = Tensor(self.data.reshape(*new_shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(old_shape)
        out._backward = _backward

        return out
    
    def sum(self):
        out = Tensor(self.data.sum(), (self,))

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward

        return out
        
    def backward(self):
        # https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        # topological order all of the children in the graph
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward() 
    
    def max(self):
        return np.max(self.data)
    
    def min(self):
        return np.min(self.data)
    
    def numpy(self):
        return np.array(self.data)
        
    @classmethod
    def zeros(cls, shape):
        assert isinstance(shape, int) or isinstance(shape, tuple), f'shape should be int or tuple insted of {type(shape)}'
        return cls(np.zeros(shape))

    @classmethod
    def ones(cls, shape):
        assert isinstance(shape, int) or isinstance(shape, tuple), f'shape should be int or tuple insted of {type(shape)}'
        return cls(np.ones(shape))
    
    @classmethod
    def normal(cls, mean=0.0, std=1.0, shape=None):
        assert isinstance(shape, int) or isinstance(shape, tuple), f'shape should be int or tuple insted of {type(shape)}'
        return cls(np.random.normal(mean, std, shape))
    
    @classmethod
    def randn(cls, *args):
        return cls(np.random.randn(*args))
    
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
        return f'Tensor(data={self.data}, dtype={self.data.dtype})'
    
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