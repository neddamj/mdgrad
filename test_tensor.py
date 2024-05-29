import numpy as np
import mdgrad
import torch

np.random.seed(42)

def test_add(op='Addition'):
    a = np.random.rand(2, 3)
    b = np.random.rand(2, 3)

    a_m = mdgrad.tensor(a)
    b_m = mdgrad.tensor(b)
    x_m = a_m + b_m
    x_m.sum().backward()

    a_t = torch.tensor(a, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    x_t = a_t + b_t
    x_t.sum().backward()

    print(op)
    print(f'mdgrad: {x_m}')
    print('\n')
    print(f'torch: {x_t}')
    print('\n\n')

    assert np.allclose(
        x_t.detach().numpy(), x_m.numpy()
    ), f'{op} results do not match'

    assert np.allclose(
        a_t.grad.detach().numpy(), a_m.grad
    ), 'Gradient results do not match'

def test_sub(op='Subtraction'):
    a = np.random.rand(2, 3)
    b = np.random.rand(2, 3)

    a_m = mdgrad.tensor(a)
    b_m = mdgrad.tensor(b)
    x_m = a_m - b_m
    x_m.sum().backward()

    a_t = torch.tensor(a, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    x_t = a_t - b_t
    x_t.sum().backward()

    print(op)
    print(f'mdgrad: {x_m}')
    print('\n')
    print(f'torch: {x_t}')
    print('\n\n')

    assert np.allclose(
        x_t.detach().numpy(), x_m.numpy()
    ), f'{op} results do not match'

    assert np.allclose(
        a_t.grad.detach().numpy(), a_m.grad
    ), 'Gradient results do not match'

def test_mul(op='Multiplication'):
    a = np.random.rand(3, 2)
    b = np.random.rand(3, 2)

    a_m = mdgrad.tensor(a)
    b_m = mdgrad.tensor(b)
    x_m = a_m * b_m
    x_m.sum().backward()

    a_t = torch.tensor(a, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    x_t = a_t * b_t
    x_t.sum().backward()

    print(op)
    print(f'mdgrad: {x_m}')
    print('\n')
    print(f'torch: {x_t}')
    print('\n\n')

    assert np.allclose(
        x_t.detach().numpy(), x_m.numpy()
    ), f'{op} results do not match'

    assert np.allclose(
        a_t.grad.detach().numpy(), a_m.grad
    ), 'Gradient results do not match'

def test_div(op='Division'):
    a = np.random.rand(3, 2)
    b = np.random.rand(3, 2)

    a_m = mdgrad.tensor(a)
    b_m = mdgrad.tensor(b)
    x_m = a_m / b_m
    x_m.sum().backward()

    a_t = torch.tensor(a, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    x_t = a_t / b_t
    x_t.sum().backward()

    print(op)
    print(f'mdgrad: {x_m}')
    print('\n')
    print(f'torch: {x_t}')
    print('\n\n')

    assert np.allclose(
        x_t.detach().numpy(), x_m.numpy()
    ), f'{op} results do not match'

    assert np.allclose(
        a_t.grad.detach().numpy(), a_m.grad
    ), 'Gradient results do not match'

def test_sum(op='Summation'):
    a = np.random.rand(3, 2)

    a_m = mdgrad.tensor(a)
    x_m = a_m.sum()
    x_m.backward()

    a_t = torch.tensor(a, requires_grad=True)
    x_t = a_t.sum()
    x_t.backward()

    print(op)
    print(f'mdgrad: {x_m}')
    print('\n')
    print(f'torch: {x_t}')
    print('\n\n')

    assert np.allclose(
        x_t.detach().numpy(), x_m.numpy()
    ), f'{op} results do not match'

    assert np.allclose(
        a_t.grad.detach().numpy(), a_m.grad
    ), 'Gradient results do not match'

def test_reshape(op='Reshape'):
    a = np.random.rand(3, 2)

    a_m = mdgrad.tensor(a)
    x_m = a_m.reshape(2, 3)
    x_m.sum().backward()

    a_t = torch.tensor(a, requires_grad=True)
    x_t = a_t.reshape(2, 3)
    x_t.sum().backward()

    print(op)
    print(f'mdgrad: {x_m}')
    print('\n')
    print(f'torch: {x_t}')
    print('\n\n')

    assert np.allclose(
        x_t.detach().numpy(), x_m.numpy()
    ), f'{op} results do not match'

    assert np.allclose(
        a_t.grad.detach().numpy(), a_m.grad
    ), 'Gradient results do not match'

def test_matmul(op='Matmul'):
    a = np.random.rand(3, 2)
    b = np.random.rand(2, 2)

    a_m = mdgrad.tensor(a)
    b_m = mdgrad.tensor(b)
    x_m = a_m @ b_m
    x_m.sum().backward()

    a_t = torch.tensor(a, requires_grad=True)
    b_t = torch.tensor(b, requires_grad=True)
    x_t = a_t @ b_t
    x_t.sum().backward()

    print(op)
    print(f'mdgrad: {x_m}')
    print('\n')
    print(f'torch: {x_t}')
    print('\n\n')

    assert np.allclose(
        x_t.detach().numpy(), x_m.numpy()
    ), f'{op} results do not match'

    assert np.allclose(
        a_t.grad.detach().numpy(), a_m.grad
    ), 'Gradient results do not match'

def test_pow(op='Exponential'):
    a = np.random.rand(3, 2)
    b = int(np.random.randint(3, 10))

    a_m = mdgrad.tensor(a)
    x_m = a_m ** b
    x_m.sum().backward()

    a_t = torch.tensor(a, requires_grad=True)
    x_t = a_t ** b
    x_t.sum().backward()

    print(op)
    print(f'mdgrad: {x_m}')
    print('\n')
    print(f'torch: {x_t}')
    print('\n\n')

    assert np.allclose(
        x_t.detach().numpy(), x_m.numpy()
    ), f'{op} results do not match'

    assert np.allclose(
        a_t.grad.detach().numpy(), a_m.grad
    ), 'Gradient results do not match'

if __name__ == '__main__':
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_sum()
    test_reshape()
    test_matmul()
    test_pow()