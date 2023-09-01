from typing import List, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from _pytest.python_api import approx

from config import REPO_ROOT
from data_processing.datasets_eval import EvalAlfredHLActionDataset
from evaluation.alignment_evaluation import calculate_alignment_metrics, _comparative_z_metrics, \
    apply_metrics_calculation_to_datasets
from grounding.models.base_models.clip import CLIPModelFrozen

from models.clasp import CLASP

Z_SIZE = 12

@pytest.fixture(scope="session")
def clip_model():
    clip: CLIPModelFrozen = MagicMock(CLIPModelFrozen)
    torch.manual_seed(42)
    clip.encode_texts = lambda x: torch.randn(len(x), 512)
    clip.encode_images = lambda x: torch.randn(x.shape[0], 512)
    yield clip

@pytest.fixture(scope="session")
def clasp_model():
    print("Instantiating CLASP model...")
    clasp: CLASP = CLASP(z_size=Z_SIZE)
    yield clasp

def test_calculate_alignment_accuracy(clasp_model, clip_model):
    # Given
    dataset = EvalAlfredHLActionDataset(REPO_ROOT / 'alfred/data/json_feat_2.1.0/valid_unseen')

    # When
    result = calculate_alignment_metrics(clasp_model, clip_model, dataset)

    # Then
    assert result['accuracy'] >= 0.
    assert result['accuracy'] <= 1.
    assert result['mrr'] >= 0.
    assert result['mrr'] <= 1.


@pytest.mark.parametrize("ground_truth,expected_accuracy, expected_mrr", [
    (0, 1., 1.),
    (1, 0., 0.5),
    (2, 0., 0.33333333333)
])
def test_comparative_z_accuracy_positive_case(ground_truth, expected_accuracy, expected_mrr):
    # Given
    z_instruction = torch.tensor([
        1, 1, 1, 0, 0, 0, 0, 0
    ])
    z_behavior_versions = torch.tensor([
        [ 1, 1, 1, 0, 0, 0, 0, 0],
        [ 1, 1, 0, 0, 0, 0, 0, 0],
        [ 1, 0, 0, 0, 0, 0, 0, 0]
    ])

    # When
    accuracy, mrr = _comparative_z_metrics(
        z_instruction,
        z_behavior_versions,
        ground_truth
    )

    # Then
    assert accuracy == expected_accuracy
    assert mrr == approx(expected_mrr)

def test_apply_metrics_calculation_to_datasets():
    # Given
    def dummy_metric_function(clasp_model, clip_model, dataset):
        return {'accuracy': 1.0, 'mrr': 0.5}
    clasp_model: CLASP = MagicMock(CLASP)
    clip_model: CLIPModelFrozen = MagicMock(CLIPModelFrozen)
    datasets: List[Tuple[str, EvalAlfredHLActionDataset]] = [
        ('valid_seen', MagicMock(EvalAlfredHLActionDataset)),
        ('valid_unseen', MagicMock(EvalAlfredHLActionDataset)),
    ]

    # When
    result: pd.DataFrame = apply_metrics_calculation_to_datasets(
        metric_function=dummy_metric_function,
        clasp_model=clasp_model,
        clip_model=clip_model,
        datasets=datasets
    )

    # Then
    expected_result = pd.DataFrame({
        'valid_seen': {'accuracy': 1.0, 'mrr': 0.5},
        'valid_unseen': {'accuracy': 1.0, 'mrr': 0.5}
    })
    print(expected_result)
    pd.testing.assert_frame_equal(expected_result, result)