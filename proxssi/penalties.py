import torch
import math

def _get_l2_norms(groups):
    norms = []
    for group in groups:
        if group.get('groups_fn') is None:
            continue

        groups_fn = group['groups_fn']
        groups_norms = []
        for param in groups_fn(group['params']):
            if not param.requires_grad:
                continue
            groups_norms.append(torch.linalg.norm(param, dim=(0, 1))
                                * math.sqrt(param.size(-1)))

        if groups_norms:
            norms.append(torch.cat(groups_norms))

    return torch.cat(norms) if norms else None


def l1_l2(groups, lambda_):
    norms = _get_l2_norms(groups)
    if norms is None:
        return 0.

    return lambda_ * torch.sum(norms)

def group_mcp(groups, lambda_, gamma):
    norms = _get_l2_norms(groups)
    if norms is None:
        return 0.

    mask = (norms <= gamma * lambda_)

    results = torch.full_like(norms, 0.5 * gamma * (lambda_ ** 2))
    results[mask] = lambda_ * norms[mask] - 0.5 * (norms[mask] ** 2) / gamma

    return torch.sum(results)
