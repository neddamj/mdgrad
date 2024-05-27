import numpy as np

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001):
        super().__init__(parameters, lr)

    def __call__(self):
        self.step()
    
    def step(self):
        for param in self.parameters:
            param.data = param.data - self.lr * param.grad