import pytest
from torch import Tensor
import torch
from evaluation.scoring_methods.forced_scoring import reciprocal_rank_with_similarity, reciprocal_rank


@pytest.mark.parametrize("losses,target_index,expected_score", [
    ([0, 0.8, 0.7, 0.99, 0.5], 0, 1),
    ([0, 0.8, 0.7, 0.99, 0.5], 3, 1./5),
    ([0, 0.8, 0.7, 0.99, 0.5], 4, 1./2),
])
def test_reciprocal_rank(losses, target_index, expected_score):
    # Given
    losses: Tensor = torch.tensor(losses)
    target_index: int = target_index

    # When
    score = reciprocal_rank(losses, target_index)

    # Then
    assert score == expected_score



@pytest.mark.parametrize("similarities,target_index,expected_score", [
    ([0, 0.8, 0.7, 0.99, 0.5], 3, 1),
    ([0, 0.8, 0.7, 0.99, 0.5], 0, 1./5),
    ([0, 0.8, 0.7, 0.99, 0.5], 4, 1./4),
])
def test_reciprocal_rank_with_similarity(similarities, target_index, expected_score):
    # Given
    similarities: Tensor = torch.tensor(similarities)
    target_index: int = target_index

    # When
    score = reciprocal_rank_with_similarity(similarities, target_index)

    # Then
    assert score == expected_score
