import pytest

import numpy as np
import torch
import math

from proxssi.tests import prox_fn
from proxssi.mcp_l2 import _newton_raphson, prox_mcp_l2_weighted


def mcp_l2(lambda_, gamma):
    def _mcp_l2(z):
        norm = torch.linalg.norm(z, dim=(0, 1))
        mask = (norm <= gamma * lambda_)

        result = z.new_full((norm.numel(),), 0.5 * gamma * (lambda_ ** 2))
        result[mask] = lambda_ * norm[mask] - 0.5 * (norm[mask] ** 2) / gamma

        return torch.sum(result)
    return _mcp_l2


def test_newton_raphson():
    x = torch.randn(3, 5).unsqueeze(0)
    weights = torch.rand(3, 5).unsqueeze(0)
    gamma, lr, lambda_ = 1.7, 1.11, 1.13

    # First condition to apply Newton-Raphson: ||x|| <= gamma * lambda
    x.mul_(0.8 * gamma * lambda_ / torch.linalg.norm(x, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x, dim=(0, 1)) <= (gamma * lambda_))

    # Second condition to apply Newton-Raphson: ||Dx|| > lr * lambda
    weights.mul_(1.2 * lr * lambda_ / torch.linalg.norm(x * weights, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x * weights, dim=(0, 1)) > (lr * lambda_))

    def theta_function_mcp(theta):
        num = (weights * x) ** 2
        tmp = gamma * weights - lr
        den = theta * tmp + lr * gamma * lambda_
        return (gamma ** 2) * torch.sum(num / (den ** 2), dim=(0, 1)) - 1

    theta = _newton_raphson(x, weights, lr, lambda_, gamma,
        atol=1e-7, rtol=1e-7, max_iters=5)
    func = theta_function_mcp(theta)

    np.testing.assert_allclose(func.abs().numpy(), 0., atol=1e-6, rtol=1e-6)


def test_prox_mcp_l2_weighted_case1():
    x = torch.randn(3, 5).unsqueeze(0)
    weights = torch.rand(3, 5).unsqueeze(0)
    gamma, lr, lambda_ = 1.7, 1.11, 1.13

    # Case 1: ||x|| <= gamma * lambda & ||Dx|| > lr * lambda
    x.mul_(0.8 * gamma * lambda_ / torch.linalg.norm(x, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x, dim=(0, 1)) <= (gamma * lambda_))

    weights.mul_(1.2 * lr * lambda_ / torch.linalg.norm(x * weights, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x * weights, dim=(0, 1)) > (lr * lambda_))
    x_0 = x.clone()

    prox = prox_mcp_l2_weighted(x, weights, lr, lambda_, gamma,
        atol=1e-7, rtol=1e-7, max_iters=100)
    prox.requires_grad_()

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())

    # Prox is the minimizer of the proximal function
    func = prox_fn(x_0, prox, weights, lr, mcp_l2(lambda_, gamma))
    func.backward()

    np.testing.assert_allclose(torch.linalg.norm(prox.grad, dim=(0, 1)).numpy(), 0.,
                               atol=1e-6)


def test_prox_mcp_l2_weighted_case2():
    x = torch.randn(3, 5).unsqueeze(0)
    weights = torch.rand(3, 5).unsqueeze(0)
    gamma, lr, lambda_ = 1.7, 1.11, 1.13

    # Case 2: ||x|| <= gamma * lambda & ||Dx|| <= lr * lambda
    x.mul_(0.8 * gamma * lambda_ / torch.linalg.norm(x, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x, dim=(0, 1)) <= (gamma * lambda_))

    weights.mul_(0.8 * lr * lambda_ / torch.linalg.norm(x * weights, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x * weights, dim=(0, 1)) <= (lr * lambda_))
    x_0 = x.clone()

    prox = prox_mcp_l2_weighted(x, weights, lr, lambda_, gamma,
        atol=1e-7, rtol=1e-7, max_iters=100)

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())
    
    np.testing.assert_array_equal(prox.numpy(), 0.)


def test_prox_mcp_l2_weighted_case3():
    x = torch.randn(3, 5).unsqueeze(0)
    weights = torch.rand(3, 5).unsqueeze(0)
    gamma, lr, lambda_ = 1.7, 1.11, 1.13

    # Case 3: ||x|| > gamma * lambda
    x.mul_(1.2 * gamma * lambda_ / torch.linalg.norm(x, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x, dim=(0, 1)) > (gamma * lambda_))
    x_0 = x.clone()

    prox = prox_mcp_l2_weighted(x, weights, lr, lambda_, gamma,
        atol=1e-7, rtol=1e-7, max_iters=100)
    prox.requires_grad_()

    # Prox leaves x unchanged
    np.testing.assert_array_equal(prox.detach().numpy(), x_0.numpy())

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())

    # Prox is the minimizer of the proximal function
    func = prox_fn(x_0, prox, weights, lr, mcp_l2(lambda_, gamma))
    func.backward()

    np.testing.assert_allclose(prox.grad.norm('fro').abs().item(), 0.,
                               atol=1e-6)


def test_prox_mcp_l2_weighted_multiple():
    x = torch.randn(3, 5).unsqueeze(0)
    weights = torch.rand(3, 5).unsqueeze(0)
    gamma, lr, lambda_ = 1.7, 1.11, 1.13

    norms = torch.linalg.norm(x, dim=(0, 1))

    # Case 1: ||x|| <= gamma * lambda & ||Dx|| > lr * lambda
    factor = torch.ones_like(norms)
    factor[[0, 3, 4]] = 0.8 * gamma * lambda_ / norms[[0, 3, 4]]
    x.mul_(factor)

    norms_weighted = torch.linalg.norm(x * weights, dim=(0, 1))

    # Case 1: ||x|| <= gamma * lambda & ||Dx|| > lr * lambda
    factor = torch.ones_like(norms_weighted)
    factor[[4]] = 1.2 * lr * lambda_ / norms_weighted[[4]]
    weights.mul_(factor)

    # Case 2: ||x|| <= gamma * lambda & ||Dx|| <= lr * lambda
    factor = torch.ones_like(norms_weighted)
    factor[[0, 3]] = 0.8 * lr * lambda_ / norms_weighted[[0, 3]]
    weights.mul_(factor)

    # Case 3: ||x|| > gamma * lambda
    factor = torch.ones_like(norms)
    factor[[1, 2]] = 1.2 * gamma * lambda_ / norms[[1, 2]]
    x.mul_(factor)

    # Check the conditions are satisfied
    norms = torch.linalg.norm(x, dim=(0, 1))
    norms_weighted = torch.linalg.norm(x * weights, dim=(0, 1))
    assert torch.all(norms[[1, 2]] > (gamma * lambda_))
    assert torch.all(norms[[0, 3, 4]] <= (gamma * lambda_))
    assert torch.all(norms_weighted[[4]] > (lr * lambda_))
    assert torch.all(norms_weighted[[0, 3]] <= (lr * lambda_))
    x_0 = x.clone()

    prox = prox_mcp_l2_weighted(x, weights, lr, lambda_, gamma,
        atol=1e-7, rtol=1e-7, max_iters=100)
    prox.requires_grad_()

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())

    # Prox is zero for indices [0, 3]
    np.testing.assert_array_equal(prox[..., [0, 3]].detach().numpy(), 0.)

    # Prox is unchanged for indices [1, 2]
    np.testing.assert_array_equal(prox[..., [1, 2]].detach().numpy(),
                                  x_0[..., [1, 2]].numpy())

    # Prox is the minimizer of the proximal function
    func = prox_fn(x_0, prox, weights, lr, mcp_l2(lambda_, gamma))
    func.backward()

    np.testing.assert_allclose(prox.grad[..., [4]].norm('fro').abs().item(), 0.,
                               atol=1e-6)


def test_newton_raphson_convolution():
    x = torch.randn(3, 5, 2, 7).view((3, 5, -1)).transpose(1, 2)
    weights = torch.rand(3, 5, 2, 7).view((3, 5, -1)).transpose(1, 2)
    gamma, lr, lambda_ = 1.7, 1.11, 1.13

    # First condition to apply Newton-Raphson: ||x|| <= gamma * lambda
    x.mul_(0.8 * gamma * lambda_ / torch.linalg.norm(x, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x, dim=(0, 1)) <= (gamma * lambda_))

    # Second condition to apply Newton-Raphson: ||Dx|| > lr * lambda
    weights.mul_(1.2 * lr * lambda_ / torch.linalg.norm(x * weights, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x * weights, dim=(0, 1)) > (lr * lambda_))

    def theta_function_mcp(theta):
        num = (weights * x) ** 2
        tmp = gamma * weights - lr
        den = theta * tmp + lr * gamma * lambda_
        return (gamma ** 2) * torch.sum(num / (den ** 2), dim=(0, 1)) - 1

    theta = _newton_raphson(x, weights, lr, lambda_, gamma,
        atol=1e-7, rtol=1e-7, max_iters=5)
    func = theta_function_mcp(theta)

    np.testing.assert_allclose(func.abs().numpy(), 0., atol=1e-6, rtol=1e-6)


def test_prox_mcp_l2_weighted_multiple_convolution():
    x = torch.randn(3, 5, 2, 7).view((3, 5, -1)).transpose(1, 2)
    weights = torch.rand(3, 5, 2, 7).view((3, 5, -1)).transpose(1, 2)
    gamma, lr, lambda_ = 1.7, 1.11, 1.13

    norms = torch.linalg.norm(x, dim=(0, 1))

    # Case 1: ||x|| <= gamma * lambda & ||Dx|| > lr * lambda
    factor = torch.ones_like(norms)
    factor[[0, 3, 4]] = 0.8 * gamma * lambda_ / norms[[0, 3, 4]]
    x.mul_(factor)

    norms_weighted = torch.linalg.norm(x * weights, dim=(0, 1))

    # Case 1: ||x|| <= gamma * lambda & ||Dx|| > lr * lambda
    factor = torch.ones_like(norms_weighted)
    factor[[4]] = 1.2 * lr * lambda_ / norms_weighted[[4]]
    weights.mul_(factor)

    # Case 2: ||x|| <= gamma * lambda & ||Dx|| <= lr * lambda
    factor = torch.ones_like(norms_weighted)
    factor[[0, 3]] = 0.8 * lr * lambda_ / norms_weighted[[0, 3]]
    weights.mul_(factor)

    # Case 3: ||x|| > gamma * lambda
    factor = torch.ones_like(norms)
    factor[[1, 2]] = 1.2 * gamma * lambda_ / norms[[1, 2]]
    x.mul_(factor)

    # Check the conditions are satisfied
    norms = torch.linalg.norm(x, dim=(0, 1))
    norms_weighted = torch.linalg.norm(x * weights, dim=(0, 1))
    assert torch.all(norms[[1, 2]] > (gamma * lambda_))
    assert torch.all(norms[[0, 3, 4]] <= (gamma * lambda_))
    assert torch.all(norms_weighted[[4]] > (lr * lambda_))
    assert torch.all(norms_weighted[[0, 3]] <= (lr * lambda_))
    x_0 = x.clone()

    prox = prox_mcp_l2_weighted(x, weights, lr, lambda_, gamma,
        atol=1e-7, rtol=1e-7, max_iters=100)
    prox.requires_grad_()

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())

    # Prox is zero for indices [0, 3]
    np.testing.assert_array_equal(prox[..., [0, 3]].detach().numpy(), 0.)

    # Prox is unchanged for indices [1, 2]
    np.testing.assert_array_equal(prox[..., [1, 2]].detach().numpy(),
                                  x_0[..., [1, 2]].numpy())

    # Prox is the minimizer of the proximal function
    func = prox_fn(x_0, prox, weights, lr, mcp_l2(lambda_, gamma))
    func.backward()

    np.testing.assert_allclose(prox.grad[..., [4]].norm('fro').abs().item(), 0.,
                               atol=1e-6)