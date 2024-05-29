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
        for p in self.parameters:
            p.data -= self.lr * p.grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.t = 0
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - (self.beta1 ** self.t))
            # Compute bias corrected second raw moment estimate
            v_hat = self.v[i] / (1 - (self.beta2 ** self.t))
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
