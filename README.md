# madnet

A small autograd engine that implements backpropagation (reverse-mode autodiff). Heavily inspired by karpathy's [micrograd](https://github.com/karpathy/micrograd/tree/master), and extended to support operations on tensors instead of scalars. Includes a small neural network api for building and training sequential neural network models.

Hopefully useful as an educational resource.

# Example Usage

```
x1 = Tensor([[8.0, 2.0]])
w1 = Tensor([[3.2, 1.2]])
z1 = w1 * x1 + x1
q1 = z1.relu()
y1 = q1 @ x1.transpose()
y1.backward()
```