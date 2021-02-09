import torch


penalties = [
    ('l1_l2', {'lambda_': 1e-5}),
    ('group_mcp', {'lambda_': 1e-5, 'gamma': 1e3})
]

def prox_fn(x, z, weights, lr, r, *args, **kwargs):
    return 0.5 * torch.sum(weights * (x - z) ** 2) + lr * r(z, *args, **kwargs)
