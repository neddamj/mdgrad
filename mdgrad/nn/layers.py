from mdgrad.tensor import Tensor, mean, max
from .utils import im2col, col2im
import numpy as np

class Module:
    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
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

class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        m, C, H, W = x.shape
        return x.reshape(m, C*H*W)
    
    def parameters(self):
        return []
    
class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        # https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast#I-Forward-propagation
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.w = Tensor.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) / np.sqrt(self.kernel_size)
        self.b = Tensor.zeros(self.out_channels)

    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        if len(x.shape) == 3:
            x = Tensor(np.expand_dims(x.data, axis=0), x._prev)
        m, n_C, n_H, n_W = x.shape
        assert self.in_channels == n_C, f'input has {n_C} channels. layer expects {self.in_channels} channels'

        C = self.out_channels
        H = int((n_H + 2 * self.padding - self.kernel_size)/ self.stride) + 1
        W = int((n_W + 2 * self.padding - self.kernel_size)/ self.stride) + 1
        X_col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.padding)
        w_col = self.w.data.reshape((self.out_channels, -1))
        b_col = self.b.data.reshape(-1, 1)
        # Perform matrix multiplication.
        out = w_col @ X_col + b_col
        # Reshape back matrix to image.
        out = Tensor(np.array(np.hsplit(out, m)).reshape((m, C, H, W)), (x,))
        #self.cache = X, X_col, w_col

        def _backward():
            #X, X_col, w_col = self.cache
            m, _, _, _ = x.shape
            # Compute bias gradient.
            dout = out.grad.copy()
            self.b.grad = np.sum(dout, axis=(0,2,3))
            # Reshape dout properly.
            dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
            dout = np.array(np.vsplit(dout, m))
            dout = np.concatenate(dout, axis=-1)
            # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
            dX_col = w_col.T @ dout
            # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
            dw_col = dout @ X_col.T
            # Reshape back to image (col2im).
            x.grad = col2im(dX_col, x.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
            # Reshape dw_col into dw.
            self.w.grad = dw_col.reshape((dw_col.shape[0], self.in_channels, self.kernel_size, self.kernel_size))
        out._backward = _backward

        return out
    
    def parameters(self):
        return [self.w, self.b]
    
class AvgPool2D(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        # https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast#I-Forward-propagation
        m, n_C, n_H, n_W = x.shape
        C = n_C
        H = int((n_H + 2 * self.padding - self.kernel_size)/ self.stride) + 1
        W = int((n_W + 2 * self.padding - self.kernel_size)/ self.stride) + 1
        X_col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.padding)
        X_col = X_col.reshape(C, X_col.shape[0]//C, -1)
        A_pool = np.mean(X_col, axis=1)
        # Reshape A_pool properly.
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, C, H, W)
        out = Tensor(A_pool, ((x,)))

        def _backward():
            #X = self.cache
            m, n_C_prev, n_H_prev, n_W_prev = x.shape

            n_C = n_C_prev
            n_H = int((n_H_prev + 2 * self.padding - self.kernel_size)/ self.stride) + 1
            n_W = int((n_W_prev + 2 * self.padding - self.kernel_size)/ self.stride) + 1
            dout = out.grad
            dout_flatten = dout.reshape(n_C, -1) / (self.kernel_size * self.kernel_size)
            dX_col = np.repeat(dout_flatten, self.kernel_size*self.kernel_size, axis=0)
            x.grad = col2im(dX_col, x.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
            # Reshape dX properly.
            x.grad = x.grad.reshape(m, -1)
            x.grad = np.array(np.hsplit(x.grad, n_C_prev))
            x.grad = x.grad.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        out._backward = _backward

        return out
        
    def parameters(self):
        return []

class Conv2DNaive(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.w = Tensor.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) / np.sqrt(self.kernel_size ** 2)
        self.b = Tensor.zeros(self.out_channels)

    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        if len(x.shape) == 3:
            x = Tensor(np.expand_dims(x.data, axis=0), x._prev)
        m, n_C, n_H, n_W = x.shape
        assert self.in_channels == n_C, f'input has {n_C} channels. layer expects {self.in_channels} channels'

        # Create a zero array in the shape of the output of the layer
        C = self.out_channels
        H = int((n_H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        W = int((n_W + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out = Tensor.zeros((m, C, H, W))
        # Do convolution
        for i in range(m):
            for c in range(C):
                for h in range(H):
                    # Slide the filter vertically
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size
                    for w in range(W):
                        # Slide the filter horizontally
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        out[i, c, h, w] = (x[i, :, h_start:h_end, w_start:w_end] * self.w[c, ...]).sum().data + self.b[c]
        # Add previous tensors to computation graph
        out._prev = set((x,))  
                     
        def _backward():
            nonlocal x
            m, C, H, W = out.shape
            for i in range(m):
                for c in range(C):
                    for h in range(H):
                        # Slide the filter vertically
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        for w in range(W):
                            # Slide the filter horizontally
                            w_start = w * self.stride
                            w_end = w_start + self.kernel_size
                            # Gradients of weights
                            self.w.grad[c, ...] += out.grad[i, c, h, w] * x[i, :, h_start:h_end, w_start:w_end]
                            # Gradients of inputs
                            x.grad[i, :, h_start:h_end, w_start:w_end] += out.grad[i, c, h, w] * self.w[c, ...]
            
            for c in range(self.out_channels):
                # Gradients of biases
                self.b.grad[c, ...] = out.grad[:, c, ...].sum()
        out._backward = _backward

        return out
    
    def parameters(self):
        return [self.w, self.b]
    

class AvgPool2DNaive(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        if len(x.shape) == 3:
            x = Tensor(np.expand_dims(x.data, axis=0), x._prev)
        m, n_C, n_H, n_W = x.shape

        C = n_C
        H = int((n_H - self.kernel_size) / self.stride) + 1
        W = int((n_W - self.kernel_size) / self.stride) + 1
        out = Tensor.zeros((m, C, H, W))
        for i in range(m):
            for c in range(C):
                for h in range(H):
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size
                    for w in range(W):
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        out[i, c, h, w] = mean(x[i, c, h_start:h_end, w_start:w_end]).data
        # Add previous tensor to computation graph
        out._prev = set((x,))

        def _backward():
            nonlocal x
            m, C, H, W = out.shape
            for i in range(m):
                for c in range(C):
                    for h in range(H):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        for w in range(W):
                            w_start = w * self.stride
                            w_end = w_start + self.kernel_size
                            # Gradient of the inputs
                            avg = out.grad[i, c, h, w] / (self.kernel_size ** 2)
                            avg *= Tensor.ones((self.kernel_size, self.kernel_size))
                            x.grad[i, c, h_start:h_end, w_start:w_end] += avg
        out._backward = _backward

        return out
    
    def parameters(self):
        return []