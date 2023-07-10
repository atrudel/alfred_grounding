from typing import Callable, List

from grounding.evaluation.scoring.generative_scoring import compute_generative_accuracy_for_action
from grounding.evaluation.scoring.forced_scoring import compute_forced_metrics_for_single_action


def get_mode_scorer(mode: str) -> Callable:
    if mode == 'forced':
        return compute_forced_metrics_for_single_action
    if mode == 'generative':
        return compute_generative_accuracy_for_action
    raise ValueError(f"Unknown scoring mode: {mode}")


def get_mode_metrics(mode: str) -> List[str]:
    if mode == 'forced':
        return ['accuracy', 'mrr']
    if mode == 'generative':
        return ['accuracy']
    raise ValueError(f"Unknown scoring mode: {mode}")
