import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, TextIO, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from grounding.evaluation.utils import load_eval_data
from grounding.data_processing.action import Action
from grounding.data_processing.datasets_eval import EvalAlfredHLActionDataset
from grounding.evaluation.scoring_methods.mode_selection import get_mode_scorer
from grounding.evaluation.utils import parse_eval_args, announce_start_evaluation
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder


device = "cuda" if torch.cuda.is_available() else "cpu"
BASELINE: str = "baseline_evaluation"


def baseline_evaluation(model: ImageConditionedLLMOnDecoder,
                        datasets: List[Tuple[str, EvalAlfredHLActionDataset]],
                        mode: str,
                        model_dir: Path) -> pd.DataFrame:
    results = {}
    with open(model_dir / f"baseline_evaluation_{mode}.log", 'w') as log_file:
        for dataset_name, dataset in datasets:
            print(f"=== {dataset_name.upper()}", "=" * 70, file=log_file)
            print(f"{dataset_name} dataset")
            scores_by_metric_and_action_type: Dict[str, Dict[str, float]] = score_object_selection_by_action_type(
                model,
                dataset,
                mode=mode,
                log_file=log_file
            )
            for metric_name, scores in scores_by_metric_and_action_type.items():
                results[f"{dataset_name}_{metric_name}"] = scores
            print("\n\n", file=log_file)

    results_df = pd.DataFrame(results)
    return results_df


def score_object_selection_by_action_type(model: ImageConditionedLLMOnDecoder,
                                          dataset: EvalAlfredHLActionDataset,
                                          mode: str,
                                          log_file: Optional[TextIO] = None) -> Dict[str, Dict[str, float]]:
    # Todo: implement the PutObject actions with two arguments
    selected_action_types: List[str] = [
        'PickupObject',
        #'OpenObject',
        #'CloseObject',
        'ToggleObject',
        'HeatObject',
        'CleanObject',
        'SliceObject',
        'CoolObject'
    ]
    actions_by_type: Dict[str, List[Action]] = dataset.get_actions_by_type()

    all_scores = defaultdict(dict)
    for action_type in selected_action_types:
        print(f"Evaluating action {action_type}")
        actions = actions_by_type[action_type]
        scores = score_object_selection_in_many_actions(model, actions, mode, log_file)
        for metric_name, score in scores.items():
            all_scores[metric_name][action_type] = score
    return all_scores


def score_object_selection_in_many_actions(model: ImageConditionedLLMOnDecoder,
                                           actions: List[Action],
                                           mode: str,
                                           log_file: Optional[TextIO] = None) -> Dict[str, float]:
    scorer = get_mode_scorer(mode)

    all_scores = defaultdict(list)
    for action in tqdm(actions):
        scores: Dict[str, float] = scorer(action, model, log_file)
        for metric_name, score in scores.items():
            all_scores[metric_name].append(score)

    # Average all scores
    averaged_scores = dict()
    for metric_name, scores in all_scores.items():
        averaged_scores[metric_name] = np.array(scores).mean()

    return averaged_scores


if __name__ == '__main__':
    description = "Baseline evaluation routine"
    args: argparse.Namespace = parse_eval_args(description)
    announce_start_evaluation(args)

    print("Loading data...")
    datasets: List[Tuple[str, EvalAlfredHLActionDataset]] = load_eval_data(args)

    print("Loading model...")
    model = ImageConditionedLLMOnDecoder.load(args.model_path).to(device)

    print("Performing baseline forced evaluation...")
    fored_results: pd.DataFrame = baseline_evaluation(model=model, datasets=datasets, mode='forced', model_dir=args.model_dir)

    print("Performing baseline generative evaluation...")
    generative_results: pd.DataFrame = baseline_evaluation(model=model, datasets=datasets, mode='generative', model_dir=args.model_dir)

    final_results: pd.DataFrame = fored_results
    for gen_column in list(generative_results):
        final_results[f"{gen_column}_generative"] = generative_results[gen_column]
    save_file: Path = args.model_dir / f"baseline_evaluation.csv"
    final_results.to_csv(save_file)
    print(f"Results saved to {save_file}")

