import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16
from dataclasses import dataclass

from proxssi.groups.vgg import vgg_groups
from proxssi.optimizers.adamw_hf import AdamW
from proxssi.tests import penalties

@dataclass
class Arguments:
    learning_rate: float = 1e-3
    weight_decay: float = 0.


@pytest.fixture
def vgg16_model():
    model = vgg16(num_classes=10)
    model.avgpool = nn.Identity()
    model.classifier = nn.Linear(512, 10)
    model._initialize_weights()
    return model


@pytest.mark.slow
@pytest.mark.parametrize('penalty,prox_kwargs', penalties)
def test_vgg16_groups_prox(vgg16_model, penalty, prox_kwargs):
    args = Arguments()
    grouped_params = vgg_groups(vgg16_model, args)

    optimizer_kwargs = {
        'lr': args.learning_rate,
        'penalty': penalty,
        'prox_kwargs': prox_kwargs
    }

    optimizer = AdamW(grouped_params, **optimizer_kwargs)

    # Dummy inputs
    inputs = torch.rand(2, 3, 32, 32)
    targets = torch.randint(10, (2,))

    optimizer.zero_grad()

    loss = F.cross_entropy(vgg16_model(inputs), targets)
    loss.backward()

    optimizer.step()
