def conv_groups_fn(params):
    return [param.view(param.shape[:2] + (-1,)).transpose(2, 1) for param in params]

def column_groups_fn(params):
    return [param.unsqueeze(0) for param in params]

def resnet_groups(model, args):
    to_prox_conv, to_prox_linear = [], []
    remaining = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.ndim == 4:
                to_prox_conv.append(param)
            elif param.ndim == 2:
                to_prox_linear.append(param)
            else:
                remaining.append(param)  # BN weight
        else:
            remaining.append(param)

    optimizer_grouped_parameters = [
        {
            'params': to_prox_conv,
            'weight_decay': args.weight_decay,
            'groups_fn': conv_groups_fn
        },
        {
            'params': to_prox_linear,
            'weight_decay': args.weight_decay,
            'groups_fn': column_groups_fn
        }
    ]
    if remaining:
        optimizer_grouped_parameters.append({
            'params': remaining,
            'weight_decay': args.weight_decay,
            'groups_fn': None
        })

    return optimizer_grouped_parameters
