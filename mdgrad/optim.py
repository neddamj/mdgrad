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
    def __init__(self, parameters, lr=0.001, nesterov=False, momentum=0, dampening=0, maximize=False):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.t = 0
        self.nesterov = nesterov
        self.maximize = maximize
        self.b = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        self.t += 1
        for i, p in enumerate(self.parameters):
            grad = p.grad
            if self.momentum != 0:
                if self.t > 0:
                    self.b[i] = self.momentum * self.b[i] + (1 - self.dampening) * grad
                else:
                    self.b[i] = grad
                
                if self.nesterov:
                    grad += self.momentum * self.b[i]
                else:
                    grad = self.b[i]
            if self.maximize:
                grad *= -1
            p.data -= self.lr * grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, maximize=False):
        super().__init__(parameters, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.t = 0
        self.eps = eps
        self.maximize = maximize
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        # https://arxiv.org/pdf/1412.6980
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        self.t += 1
        for i, p in enumerate(self.parameters):
            grad = p.grad
            if self.maximize:
                grad *= -1
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - (self.beta1 ** self.t))
            # Compute bias corrected second raw moment estimate
            v_hat = self.v[i] / (1 - (self.beta2 ** self.t))
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
