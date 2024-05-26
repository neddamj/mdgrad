# mdgrad

A small autograd engine that implements backpropagation (reverse-mode autodiff). Heavily inspired by karpathy's [micrograd](https://github.com/karpathy/micrograd/tree/master), and extended to support operations on tensors instead of scalars. Includes a small neural network api for building and training neural networks.

Hopefully useful as an educational resource.

## Installation

``` bash
pip install mdgrad
```

## Example Usage

A silly example showing supported operations

```python

import mdgrad
import mdgrad.nn as nn

a = 3 * mdgrad.randn(3, 2)
b = mdgrad.ones(shape=(2, 2))
c = a @ b
d = c * 3 / 2
e = d ** 2
f = e.sum()
print(f.data) 
f.backward()
print(a.grad) 
```

An example showing how to define and run a neural network. See demo.ipynb for more details on training.

```python

import mdgrad
import mdgrad.nn as nn

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
X = mdgrad.randn(100, 2)
target = mdgrad.randn(100, 1)

# Compute output and loss
out = model(X)
loss = loss_fn(out, target)

# Compute gradients of parameters
loss.backward()
```
