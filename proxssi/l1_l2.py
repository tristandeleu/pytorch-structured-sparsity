import torch
import torch.nn.functional as F

from math import sqrt
from torch import Tensor
from torch.jit import Future
from typing import Tuple, List, Iterable


@torch.jit.script
def prox_l2(param: Tensor,
            lambda_: float) -> Tensor:
    norm = F.relu(torch.linalg.norm(param, dim=(0, 1)) - lambda_)
    norm.div_(norm + lambda_)

    param.data.mul_(norm)
    return param

@torch.jit.script
def prox_l1_l2(groups: List[Tensor],
               lr: float,
               lambda_: float,
               reweight: bool = True):
    futures: List[Future[Tensor]] = []

    if reweight:
        lambdas: List[float] = [sqrt(param.numel() / param.size(-1)) * lambda_
                                for param in groups]
    else:
        lambdas: List[float] = [lambda_ for _ in groups]

    for param, lambda_rw in zip(groups, lambdas):
        futures.append(torch.jit.fork(prox_l2, param, lr * lambda_rw))

    for future in futures:
        torch.jit.wait(future)


@torch.jit.script
def _newton_raphson_step(theta: Tensor,
                         weights: Tensor,
                         num: Tensor,
                         lambda_: float) -> Tensor:
    den: Tensor = theta + lambda_ / weights
    func: Tensor = torch.sum(num / (den ** 2), dim=(0, 1)) - 1
    step: Tensor = 0.5 * (func / torch.sum(num / (den ** 3), dim=(0, 1)))

    theta.add_(step).clamp(min=0.)
    return func

@torch.jit.script
def _newton_raphson(param: Tensor,
                    weights: Tensor,
                    lambda_: float,
                    atol: float = 1e-7,
                    rtol: float = 1e-7,
                    max_iters: int = 100) -> Tensor:
    num_groups: int = param.size(-1)

    d_max: Tensor = torch.max(weights.reshape((-1, num_groups)), dim=0)[0]
    norms_weighted: Tensor = torch.linalg.norm(param * weights, dim=(0, 1))

    theta: Tensor = (norms_weighted - lambda_) / d_max
    num: Tensor = param ** 2

    prev_value: Tensor = param.new_zeros((num_groups,))
    for _ in range(max_iters):
        value: Tensor = _newton_raphson_step(theta, weights, num, lambda_)
        if torch.all(value.abs() < atol):
            break
        if torch.all((prev_value - value).abs() < rtol):
            break
        prev_value = value

    return theta

@torch.jit.script
def prox_l2_weighted(param: Tensor,
                     weights: Tensor,
                     lambda_: float,
                     atol: float = 1e-7,
                     rtol: float = 1e-7,
                     max_iters: int = 100) -> Tensor:
    mask: Tensor = (torch.linalg.norm(param * weights, dim=(0, 1)) > lambda_)
    if torch.any(mask):
        theta: Tensor = _newton_raphson(param[:,:,mask], weights[:,:,mask], lambda_,
                                        atol=atol, rtol=rtol, max_iters=max_iters)

        factor: Tensor = torch.zeros_like(param)
        theta_weights: Tensor = weights[:,:,mask] * theta
        factor[:,:,mask] = theta_weights / (theta_weights + lambda_)

        param.data.mul_(factor)
    else:
        param.data.zero_()
    return param

@torch.jit.script
def prox_l1_l2_weighted(groups: List[Tensor],
                        weights: List[Tensor],
                        lr: float,
                        lambda_: float,
                        atol: float = 1e-7,
                        rtol: float = 1e-7,
                        max_iters: int = 100,
                        reweight: bool = True):
    futures: List[Future[Tensor]] = []

    if reweight:
        lambdas: List[float] = [sqrt(param.numel() / param.size(-1)) * lambda_
                                for param in groups]
    else:
        lambdas: List[float] = [lambda_ for _ in groups]

    for param, weight, lambda_rw in zip(groups, weights, lambdas):
        futures.append(torch.jit.fork(prox_l2_weighted, param,
            weight, lr * lambda_rw, atol=atol, rtol=rtol, max_iters=max_iters))

    for future in futures:
        torch.jit.wait(future)
