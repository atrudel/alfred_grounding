import unittest.mock
from unittest.mock import Mock

from grounding.data_processing.action import Action
from grounding.data_processing.datasets_eval import EvalAlfredHLActionDataset
from grounding.evaluation.clasp.behavior_generation_evaluation import evaluate_object_generative_accuracy_in_behavior_generation, \
    evaluate_object_forced_metrics_in_behavior_generation
from grounding.models.clasp import CLASP


def test_evaluate_object_forced_metrics_in_behavior_generation():
    # Given
    def mock_get_testable_actions():
        actions = []
        for i in range(20):
            action = Mock(Action)
            action.fake_id = i
            actions.append(action)
        return actions

    class_to_mock = "grounding.evaluation.clasp.behavior_generation_evaluation.compute_forced_metrics_for_single_action"
    def mock_compute_forced_metrics_for_single_action(action, model):
        return {
            'metric_1': action.fake_id,
            'metric_2': 0.5
        }
    mock_dataset: EvalAlfredHLActionDataset = Mock(EvalAlfredHLActionDataset)
    mock_dataset.get_testable_actions = mock_get_testable_actions
    mock_clasp: CLASP = Mock(CLASP)

    # When
    with unittest.mock.patch(class_to_mock, mock_compute_forced_metrics_for_single_action):
        result = evaluate_object_forced_metrics_in_behavior_generation(mock_dataset, mock_clasp)

    # Then
    expected_result = {
        'metric_1': 9.5,
        'metric_2': 0.5
    }
    assert result == expected_result