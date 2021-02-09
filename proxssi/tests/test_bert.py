import pytest
import torch

from transformers import BertForQuestionAnswering, BertConfig
from transformers import TrainingArguments

from proxssi.groups.bert import bert_groups
from proxssi.optimizers.adamw_hf import AdamW
from proxssi.tests import penalties


@pytest.fixture
def bert_model():
    config = BertConfig()
    model = BertForQuestionAnswering(config=config)
    return model


@pytest.mark.slow
@pytest.mark.parametrize('penalty,prox_kwargs', penalties)
def test_bert_groups_prox(bert_model, penalty, prox_kwargs):
    args = TrainingArguments('output_dir')
    grouped_params = bert_groups(bert_model, args)

    optimizer_kwargs = {
        'lr': args.learning_rate,
        'betas': (args.adam_beta1, args.adam_beta2),
        'eps': args.adam_epsilon,
        'penalty': penalty,
        'prox_kwargs': prox_kwargs
    }

    optimizer = AdamW(grouped_params, **optimizer_kwargs)

    # Dummy inputs
    input_ids = torch.randint(bert_model.config.vocab_size, (2, 3))
    token_type_ids = torch.randint(bert_model.config.type_vocab_size, (2, 3))
    start_positions = torch.tensor([0, 1], dtype=torch.long)
    end_positions = torch.tensor([2, 2], dtype=torch.long)

    optimizer.zero_grad()

    outputs = bert_model(input_ids=input_ids, token_type_ids=token_type_ids,
        start_positions=start_positions, end_positions=end_positions)
    outputs.loss.backward()

    optimizer.step()
