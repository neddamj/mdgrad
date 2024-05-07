import numpy as np

class Tensor:
    def __init__(self, data, _children=()):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        # Set data to floats to division can be done
        self.data = self.data.astype(float) 
        self._prev = set(_children)
        self.grad = np.zeros_like(self.data)
        self.shape = self.data.shape
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Tensor(data={self.data})'
    
    def __add__(self, other):
        # Elementwise addition. Tensors must be the same size, or one of 
        # them must be a scalar 
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        # Elementwise multiplication. Tensors must be the same size, or one of 
        # them must be a scalar 
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        # The exponent must be a scalar
        assert isinstance(other, (int, float)), "Exponent must be a scalar (int/float)"
        out = Tensor(self.data ** other, (self, other))

        def _backward():
            self.grad += (other * self.data ** (other -1)) * out.grad
        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        # Rows of self must match columns of other
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ np.transpose(other.data)
            other.grad += np.transpose(out.grad @ self.data)
        out._backward = _backward

        return out
    
    def relu(self):
        out = self.data * (self.data > 0)
        out = Tensor(self.data * (self.data > 0), (self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def transpose(self):
        out = Tensor(np.transpose(self.data), (self,))
        
        def _backward():
            self.grad += self.grad
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

    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):    
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1