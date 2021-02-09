def column_groups_fn(params):
    return [param.unsqueeze(0) for param in params]

def bert_groups(model, args):
    not_to_decay, remaining = [], []
    to_prox_columns = []

    for name, param in model.named_parameters():
        if any(nd in name for nd in ['bias', 'LayerNorm.weight']):
            not_to_decay.append(param)
        elif any(nd in name for nd in ['weight']):
            to_prox_columns.append(param)
        else:
            remaining.append(param)

    optimizer_grouped_parameters = [
        {
            'params': to_prox_columns,
            'weight_decay': args.weight_decay,
            'groups_fn': column_groups_fn
        },
        {
            'params': not_to_decay,
            'weight_decay': 0.0,
            'groups_fn': None
        },
    ]
    if remaining:
        optimizer_grouped_parameters.append({
            'params': remaining,
            'weight_decay': args.weight_decay,
            'groups_fn': None
        })

    return optimizer_grouped_parameters
