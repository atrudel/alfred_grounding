import pickle
from typing import List

import pytest
import torch
from unittest.mock import patch

from torch import Tensor

from config import REPO_ROOT
from models.clasp import CLASP

Z_SIZE = 4

@pytest.fixture(scope="session")
def clasp_model():
    # Instantiate your object here
    clasp: CLASP = CLASP(z_size=Z_SIZE)
    yield clasp


def test_contrastive_loss_should_return_zero_with_identical_zs(clasp_model: CLASP):
    # Given
    z_text = torch.tensor([[1,0,0,0],
                           [0,0,1,0],
                           [0,1,0,0]], dtype=float)
    z_behavior = z_text.clone()

    # When
    loss = clasp_model.contrastive_loss(z_text, z_behavior)

    # Then
    assert loss.item() == pytest.approx(0, abs=1e-5)


def test_contrastive_loss_should_be_high_with_orthogonal_z(clasp_model: CLASP):
    # Given
    z_size = 8
    z_text = torch.tensor([[1, 0, 0, 0 ],
                           [0, 0, 1, 0]], dtype=float)
    z_behavior = torch.tensor([[0, 1, 0, 0 ],
                               [0, 0, 0, 1]], dtype=float)

    # When
    loss = clasp_model.contrastive_loss(z_text, z_behavior)

    # Then
    assert loss > 0.5


def test_evaluate_candidate_commands_on_all_object_options(clasp_model: CLASP):
    # Given
    action_file = REPO_ROOT / "alfred/data/json_feat_2.1.0/valid_seen/preprocessed_actions/Act_VSEEN_000001_PickupObject(['butterknife']).pickle"
    with open(action_file, 'rb') as f:
        action = pickle.load(f)
        candidate_output_texts: List[str] = action.make_command_options_for_all_objects()

    # When
    logits, output_toks = clasp_model.evaluate_candidate_commands_on_all_object_options(action, candidate_output_texts)

    # Then
    assert type(logits) == Tensor
    assert type(output_toks) == Tensor
    assert logits.shape == (80, 20, clasp_model.behavior_generator.gpt.tokenizer.vocab_size)
    assert output_toks.shape == (80, 20)


# def test_evaluate_candidate_commands_on_all_object_options_computes_correctly(clasp_model):
#     # Given
#     action_file = REPO_ROOT / "alfred/data/json_feat_2.1.0/valid_seen/preprocessed_actions/Act_VSEEN_000001_PickupObject(['butterknife']).pickle"
#     with open(action_file, 'rb') as f:
#         action = pickle.load(f)
#         candidate_output_texts: List[str] = action.make_command_options_for_all_objects()
#     # When
#     # with patch('CLASP.behavior_generator', return_value=)
#     #