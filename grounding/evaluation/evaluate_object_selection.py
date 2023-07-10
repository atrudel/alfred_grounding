import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from grounding.data_processing.action import Action, UnaccomplishedSubstitutionException
from grounding.data_processing.datasets import EvalAlfredHLActionDataset
from grounding.evaluation.scoring.forced_scoring import compute_forced_metrics_for_single_action
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Evaluation on object selection.')
parser.add_argument('--model_dir', type=str, help='Directory where the model checkpoint is located.')


def evaluate_object_substitution_by_object(model: ImageConditionedLLMOnDecoder,
                                           actions: List[Action],
                                           substitutions: dict,
                                           scorer: Callable) -> Tuple[Dict[str, float], Dict[str, float]]:
    subst_types = ['no_modif', 'sibling', 'generic', 'description', 'unrelated']
    accuracies = {subst_type: [] for subst_type in subst_types}
    mrrs = {subst_type: [] for subst_type in subst_types}
    for action in tqdm(actions):
        for subst_type in subst_types:
            try:
                if subst_type == 'no_modif':
                    subst_action = action
                else:
                    subst_action = action.make_substitution(substitutions[subst_type])
                accuracy, mrr = scorer(subst_action, model, None)
                accuracies[subst_type].append(accuracy)
                mrrs[subst_type].append(mrr)
            except UnaccomplishedSubstitutionException as e:
                print(f"Skipping example: {e}")
    for key in subst_types:
        accuracies[key] = np.array(accuracies[key]).mean()
        mrrs[key] = np.array(mrrs[key]).mean()
    return accuracies, mrrs


def fork_substitution_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir):
    substitutions = {
        'no_modif': 'fork',
        'sibling': 'knife',
        'generic': 'utensil',
        'description': 'metal object',
        'unrelated': 'baseball'
    }
    train_forks = train_dataset.get_actions_by_objects()['fork']
    valid_seen_forks = valid_seen_dataset.get_actions_by_objects()['fork']
    valid_unseen_forks = valid_unseen_dataset.get_actions_by_objects()['fork']
    train_acc, train_mrr = evaluate_object_substitution_by_object(model, train_forks, substitutions,
                                                                  compute_forced_metrics_for_single_action)
    valid_seen_acc, valid_seen_mrr = evaluate_object_substitution_by_object(model, valid_seen_forks, substitutions,
                                                                            compute_forced_metrics_for_single_action)
    valid_unseen_acc, valid_unseen_mrr = evaluate_object_substitution_by_object(model, valid_unseen_forks, substitutions,
                                                                                compute_forced_metrics_for_single_action)
    result_df = pd.DataFrame({
        'train_acc': train_acc,
        'valid_seen_acc': valid_seen_acc,
        'valid_unseen_acc': valid_unseen_acc,
        'train_mrr': train_mrr,
        'valid_seen_mrr': valid_seen_mrr,
        'valid_unseen_mrr': valid_unseen_mrr
    })
    result_df.to_csv(model_dir / "fork_substitutions.csv")


if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args()
    model_dir: Path = Path(args.model_dir)
    model_path = model_dir / "checkpoint.pth.tar"

    model = ImageConditionedLLMOnDecoder.load(model_path).to(device)

    train_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/train')
    valid_seen_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_seen')
    valid_unseen_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_unseen')

    # baseline_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir)

    fork_substitution_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir)