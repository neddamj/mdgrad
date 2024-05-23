# madnet

A small autograd engine that implements backpropagation (reverse-mode autodiff). Heavily inspired by karpathy's [micrograd](https://github.com/karpathy/micrograd/tree/master), and extended to support operations on tensors instead of scalars. Includes a small neural network api for building and training sequential neural network models.

Hopefully useful as an educational resource.

# Installation
```
pip install madnet
```

# Example Usage

A silly example showing supported operations
```python
import madnet as mn
import madnet.nn as nn

a = mn.Tensor([[-2.0, 4.0, 5.0], 
           [1.0, -5.5, 2.4]])
b = mn.Tensor([[-2.0, 4.0], 
           [1.0, 2.4],
           [1.4, 9.0]])
c = a @ b
d = c * 3 / 2
e = d ** 2
f = e.sum()
print(f.data) # prints 5776.783
f.backward()
print(a.grad) # prints (2, 3) array which is the numerical value of df/da
```


An example showing how to define and run a neural network. See demo.ipynb for more details on training.
```python
import madnet as mn
import madnet.nn as nn

# Define the model and loss function
model = nn.Sequential([
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 50), 
    nn.ReLU(),
    nn.Linear(50, 15),
    nn.ReLU(),
    nn.Linear(15, 1),
    nn.Sigmoid()
])
loss_fn = nn.MSELoss()

# Create dummy data
X = mn.Tensor.randn(100, 2)
target = mn.Tensor.randn(100, 1)

# Compute output and loss
out = model(X)
loss = loss_fn(out, target)

# Compute gradients of parameters
loss.backward()
```