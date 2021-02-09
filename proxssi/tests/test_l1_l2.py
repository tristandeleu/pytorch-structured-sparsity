import pytest

import numpy as np
import torch
import math

from proxssi.tests import prox_fn
from proxssi.l1_l2 import _newton_raphson, prox_l2_weighted


def l2(lambda_):
    def _l2(z):
        norm = torch.linalg.norm(z, dim=(0, 1))
        return lambda_ * torch.sum(norm)
    return _l2


def test_newton_raphson():
    x = torch.randn(3, 5).unsqueeze(0)
    weights = torch.rand(3, 5).unsqueeze(0)
    lambda_ = 1.7

    # Condition to apply Newton-Raphson: ||Dx|| > lambda
    x.mul_(1.2 * lambda_ / torch.linalg.norm(x * weights, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x * weights, dim=(0, 1)) > lambda_)

    def theta_function(theta):
        num, den = x ** 2, theta + lambda_ / weights
        return torch.sum(num / (den ** 2), dim=(0, 1)) - 1

    theta = _newton_raphson(x, weights, lambda_, atol=1e-7, rtol=1e-7, max_iters=100)
    func = theta_function(theta)

    np.testing.assert_allclose(func.abs().numpy(), 0., atol=1e-6, rtol=1e-6)


def test_prox_l2_weighted_case1():
    x = torch.randn(3, 5).unsqueeze(0)
    weights = torch.rand(3, 5).unsqueeze(0)
    lr, lambda_ = 1.7, 1.11

    # Case 1: ||Dx|| > lr * lambda
    x.mul_(1.2 * lambda_ * lr / torch.linalg.norm(x * weights, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x * weights, dim=(0, 1)) > (lambda_ * lr))
    x_0 = x.clone()

    prox = prox_l2_weighted(x, weights, lr * lambda_,
        atol=1e-7, rtol=1e-7, max_iters=100)
    prox.requires_grad_()

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())

    # Prox is the minimizer of the proximal function
    func = prox_fn(x_0, prox, weights, lr, l2(lambda_))
    func.backward()

    np.testing.assert_allclose(prox.grad.norm('fro').item(), 0.,
                               atol=1e-6)


def test_prox_l2_weighted_case2():
    x = torch.randn(3, 5)
    weights = torch.rand(3, 5)
    lr, lambda_ = 1.7, 1.11

    # Case 2: ||Dx|| <= lr * lambda
    x.mul_(0.8 * lambda_ * lr / torch.linalg.norm(x * weights, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x * weights, dim=(0, 1)) <= (lambda_ * lr))

    prox = prox_l2_weighted(x, weights, lr * lambda_,
        atol=1e-7, rtol=1e-7, max_iters=100)

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())

    # Prox is 0
    np.testing.assert_array_equal(prox.numpy(), 0.)


def test_prox_l2_weighted_multiple():
    x = torch.randn(3, 5).unsqueeze(0)
    weights = torch.rand(3, 5).unsqueeze(0)
    lr, lambda_ = 1.7, 1.11

    norms = torch.linalg.norm(x * weights, dim=(0, 1))

    # Case 1: ||Dx|| > lr * lambda
    factor = torch.ones_like(norms)
    factor[[1, 2]] = 1.2 * lambda_ * lr / norms[[1, 2]]
    x.mul_(factor)

    # Case 2: ||Dx|| <= lr * lambda
    factor = torch.ones_like(norms)
    factor[[0, 3, 4]] = 0.8 * lambda_ * lr / norms[[0, 3, 4]]
    x.mul_(factor)

    # Check the conditions are satisfied
    norms = torch.linalg.norm(x * weights, dim=(0, 1))
    assert torch.all(norms[[1, 2]] > (lr * lambda_))
    assert torch.all(norms[[0, 3, 4]] <= (lr * lambda_))
    x_0 = x.clone()

    prox = prox_l2_weighted(x, weights, lr * lambda_,
        atol=1e-7, rtol=1e-7, max_iters=100)
    prox.requires_grad_()

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())

    # Prox is zero for indices [0, 3, 4]
    np.testing.assert_array_equal(prox[:,:,[0, 3, 4]].detach().numpy(), 0.)

    # Prox is the minimizer of the proximal function
    func = prox_fn(x_0, prox, weights, lr, l2(lambda_))
    func.backward()

    np.testing.assert_allclose(prox.grad[:,:,[1, 2]].norm('fro').item(), 0.,
                               atol=1e-6)


def test_newton_raphson_convolution():
    x = torch.randn(3, 5, 2, 7).view((3, 5, -1)).transpose(1, 2)
    weights = torch.rand(3, 5, 2, 7).view((3, 5, -1)).transpose(1, 2)
    lambda_ = 1.7

    # Condition to apply Newton-Raphson: ||Dx|| > lambda
    x.mul_(1.2 * lambda_ / torch.linalg.norm(x * weights, dim=(0, 1)))
    assert torch.all(torch.linalg.norm(x * weights, dim=(0, 1)) > lambda_)

    def theta_function(theta):
        num, den = x ** 2, theta + lambda_ / weights
        return torch.sum(num / (den ** 2), dim=(0, 1)) - 1

    theta = _newton_raphson(x, weights, lambda_, atol=1e-7, rtol=1e-7, max_iters=100)
    func = theta_function(theta)

    np.testing.assert_allclose(func.abs().numpy(), 0., atol=1e-6, rtol=1e-6)


def test_prox_l2_weighted_multiple_convolution():
    x = torch.randn(3, 5, 2, 7).view((3, 5, -1)).transpose(1, 2)
    weights = torch.rand(3, 5, 2, 7).view((3, 5, -1)).transpose(1, 2)
    lr, lambda_ = 1.7, 1.11

    norms = torch.linalg.norm(x * weights, dim=(0, 1))

    # Case 1: ||Dx|| > lr * lambda
    factor = torch.ones_like(norms)
    factor[[1, 2]] = 1.2 * lambda_ * lr / norms[[1, 2]]
    x.mul_(factor)

    # Case 2: ||Dx|| <= lr * lambda
    factor = torch.ones_like(norms)
    factor[[0, 3, 4]] = 0.8 * lambda_ * lr / norms[[0, 3, 4]]
    x.mul_(factor)

    # Check the conditions are satisfied
    norms = torch.linalg.norm(x * weights, dim=(0, 1))
    assert torch.all(norms[[1, 2]] > (lr * lambda_))
    assert torch.all(norms[[0, 3, 4]] <= (lr * lambda_))
    x_0 = x.clone()

    prox = prox_l2_weighted(x, weights, lr * lambda_,
        atol=1e-7, rtol=1e-7, max_iters=100)
    prox.requires_grad_()

    # Inplace changes
    np.testing.assert_array_equal(x.detach().numpy(), prox.detach().numpy())

    # Prox is zero for indices [0, 3, 4]
    np.testing.assert_array_equal(prox[..., [0, 3, 4]].detach().numpy(), 0.)

    # Prox is the minimizer of the proximal function
    func = prox_fn(x_0, prox, weights, lr, l2(lambda_))
    func.backward()

    np.testing.assert_allclose(prox.grad[..., [1, 2]].norm('fro').item(), 0.,
                               atol=1e-6)
