import torch
import torch.nn.functional as F

from math import sqrt
from torch import Tensor
from torch.jit import Future
from typing import Tuple, List, Iterable


@torch.jit.script
def prox_mcp_l2(param: Tensor,
                lr: float,
                lambda_: float,
                gamma: float) -> Tensor:
    norm = F.relu(torch.linalg.norm(param, dim=(0, 1)) - lr * lambda_)
    norm.div_(norm + lr * lambda_).mul_(gamma / (gamma - lr)).clamp_(max=1.)

    param.data.mul_(norm)
    return param

@torch.jit.script
def prox_group_mcp_l2(groups: List[Tensor],
                      lr: float,
                      lambda_: float,
                      gamma: float,
                      reweight: bool = True):
    futures: List[Future[Tensor]] = []

    if reweight:
        lambdas: List[float] = [sqrt(param.numel() / param.size(-1)) * lambda_
                                for param in groups]
    else:
        lambdas: List[float] = [lambda_ for _ in groups]

    for param, lambda_rw in zip(groups, lambdas):
        futures.append(torch.jit.fork(prox_mcp_l2,
            param, lr, lambda_rw, gamma))

    for future in futures:
        torch.jit.wait(future)


@torch.jit.script
def _newton_raphson_step(theta: Tensor,
                         num: Tensor,
                         tmp: Tensor,
                         lr: float,
                         gamma: float,
                         lambda_: float) -> Tensor:
    den: Tensor = theta * tmp + lr * gamma * lambda_
    func: Tensor = (gamma ** 2) * torch.sum(num / (den ** 2), dim=(0, 1)) - 1
    funcp: Tensor = (gamma ** 2) * torch.sum((num * tmp) / (den ** 3), dim=(0, 1))

    theta.add_(0.5 * (func / funcp)).clamp_(min=0.)  # F.relu(theta + step)
    return func

@torch.jit.script
def _newton_raphson(param: Tensor,
                    weights: Tensor,
                    lr: float,
                    lambda_: float,
                    gamma: float,
                    atol: float = 1e-7,
                    rtol: float = 1e-7,
                    max_iters: int = 100) -> Tensor:
    num_groups: int = param.size(-1)

    d_max: Tensor = torch.max(weights.reshape((-1, num_groups)), dim=0)[0]
    norms_weighted: Tensor = torch.linalg.norm(param * weights, dim=(0, 1))

    theta: Tensor = gamma * (norms_weighted - lr * lambda_) / (d_max * gamma - lr)
    num: Tensor = (weights * param) ** 2
    tmp: Tensor = gamma * weights - lr

    prev_value: Tensor = param.new_zeros((num_groups,))
    for _ in range(max_iters):
        value: Tensor = _newton_raphson_step(theta, num, tmp, lr, gamma, lambda_)
        if torch.all(value.abs() < atol):
            break
        if torch.all((prev_value - value).abs() < rtol):
            break
        prev_value = value

    return theta

@torch.jit.script
def prox_mcp_l2_weighted(param: Tensor,
                         weights: Tensor,
                         lr: float,
                         lambda_: float,
                         gamma: float,
                         atol: float = 1e-7,
                         rtol: float = 1e-7,
                         max_iters: int = 100) -> Tensor:
    first_mask: Tensor = (torch.linalg.norm(param, dim=(0, 1)) <= gamma * lambda_)
    if torch.any(first_mask):
        second_mask: Tensor = (torch.linalg.norm(param * weights, dim=(0, 1)) > lr * lambda_)

        mask: Tensor = torch.logical_and(first_mask, second_mask)
        factor: Tensor = torch.zeros_like(param)
        factor[:,:,~first_mask] = 1.
        if torch.any(mask):
            theta: Tensor = _newton_raphson(param[:,:,mask], weights[:,:,mask], lr,
                                            lambda_, gamma, atol=atol,
                                            rtol=rtol, max_iters=max_iters)

            theta_weights: Tensor = gamma * theta * weights[:,:,mask]
            tmp: Tensor = lr * (gamma * lambda_ - theta)
            factor[:,:,mask] = theta_weights / (theta_weights + tmp)

        param.data.mul_(factor)
    return param

@torch.jit.script
def prox_group_mcp_weighted(groups: List[Tensor],
                            weights: List[Tensor],
                            lr: float,
                            lambda_: float,
                            gamma: float,
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
        futures.append(torch.jit.fork(prox_mcp_l2_weighted,
            param, weight, lr, lambda_rw, gamma,
            atol=atol, rtol=rtol, max_iters=max_iters))

    for future in futures:
        torch.jit.wait(future)
