import argparse

import pandas as pd
import torch

from grounding.data_processing.datasets_eval import EvalAlfredHLActionDataset
from grounding.evaluation.evaluate_object_selection import evaluate_object_substitution_by_object
from grounding.evaluation.baseline_basic_evaluation import score_object_selection_by_action_type
from grounding.evaluation.scoring_methods.generative_scoring import compute_generative_accuracy_for_action
from grounding.evaluation.utils import parse_eval_args
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def generative_baseline_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir):
    with open(model_dir / "generative_baseline_evaluation_by_action_type.log", 'w') as log_file:
        if train_dataset is not None:
            print(f"=== TRAIN ", "=" * 70, file=log_file)
            train_acc, _ = score_object_selection_by_action_type(
               model,
               train_dataset,
                compute_generative_accuracy_for_action,
               log_file
            )
        print(f"\n\n=== VALID_SEEN ", "=" * 65, file=log_file)
        valid_seen_acc, _ = score_object_selection_by_action_type(
            model,
            valid_seen_dataset,
            compute_generative_accuracy_for_action,
            log_file
        )
        print(f"\n\n=== VALID_UNSEEN ", "=" * 65, file=log_file)
        valid_unseen_acc, _ = score_object_selection_by_action_type(
            model,
            valid_unseen_dataset,
            compute_generative_accuracy_for_action,
            log_file
        )
    results_df = pd.DataFrame({
        'valid_seen_accuracy': valid_seen_acc,
        'valid_unseen_accuracy': valid_unseen_acc,
    })
    if train_dataset is not None:
        results_df['train_accuracy'] = train_acc

    results_df.to_csv(model_dir / "generative_baseline_evaluation_by_action_type.csv")


def generative_fork_substitution_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir):
    substitutions = {
        'no_modif': 'fork',
        'sibling': 'knife',
        'generic': 'utensil',
        'description': 'metal object',
        'unrelated': 'baseball'
    }
#    train_forks = train_dataset.get_actions_by_objects()['fork']
    valid_seen_forks = valid_seen_dataset.get_actions_by_objects()['fork']
    valid_unseen_forks = valid_unseen_dataset.get_actions_by_objects()['fork']

 #   train_acc, train_mrr = evaluate_object_substitution_by_object(model, train_forks, substitutions,
 #                                                      compute_generative_metrics_for_action)
    valid_seen_acc, valid_seen_mrr = evaluate_object_substitution_by_object(model, valid_seen_forks, substitutions,
                                                                            compute_generative_accuracy_for_action)
    valid_unseen_acc, valid_unseen_mrr = evaluate_object_substitution_by_object(model, valid_unseen_forks, substitutions,
                                                                                compute_generative_accuracy_for_action)

    result_df = pd.DataFrame({
 #       'train_acc': train_acc,
        'valid_seen_acc': valid_seen_acc,
        'valid_unseen_acc': valid_unseen_acc,
#      'train_mrr': train_mrr,
        'valid_seen_mrr': valid_seen_mrr,
        'valid_unseen_mrr': valid_unseen_mrr
    })
    result_df.to_csv(model_dir / "generative_fork_substitutions.csv")


if __name__ == '__main__':
    args: argparse.Namespace = parse_eval_args(description="Evaluation of model's generation.")

    model = ImageConditionedLLMOnDecoder.load(args.model_path).to(device)

    train_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/train') if args.train else None
    valid_seen_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_seen')
    valid_unseen_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_unseen')

    generative_baseline_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, args.model_dir)
    # generative_fork_substitution_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir)
