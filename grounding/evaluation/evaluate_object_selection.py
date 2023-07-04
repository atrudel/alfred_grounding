import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Callable, TextIO, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm
from transformers import BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput

from grounding.data_processing.action import Action, UnaccomplishedSubstitutionException
from grounding.data_processing.datasets import EvalAlfredHLActionDataset
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Evaluation on object selection.')
parser.add_argument('--model_dir', type=str, help='Directory where the model checkpoint is located.')


def reciprocal_rank(losses: Tensor, target_index) -> float:
    _, indices = torch.sort(losses)
    rank = torch.where(indices == target_index)[0].item() + 1
    return 1. / rank


def compute_object_accuracy_for_same_action_type(model: ImageConditionedLLMOnDecoder,
                                                 actions: List[Action],
                                                 scorer: Callable,
                                                 log_file: Optional[TextIO] = None):
    accuracies = []
    mrrs = []
    for action in tqdm(actions):
        accuracy, mrr = scorer(action, model, log_file)
        accuracies.append(accuracy)
        mrrs.append(mrr)

    return np.array(accuracies).mean(), np.array(mrrs).mean()


def compute_forced_metrics_for_action(action: Action,
                                      model: ImageConditionedLLMOnDecoder,
                                      log_file: Optional[TextIO]) -> Tuple[float, float]:
    candidate_output_texts: List[str] = action.make_classification_strings()
    n_candidates = len(candidate_output_texts)
    input_tokenized: BatchEncoding = model.tokenizer(
        [action.instruction] * n_candidates,
        return_tensors='pt'
    )
    image_features: Tensor = action.image_features.unsqueeze(0).repeat(n_candidates, 1, 1)
    decoder_input_toks, decoder_input_att_mask, decoder_image_features, output_toks = model.prepare_image_and_output_data(
        image_features, candidate_output_texts
    )
    with torch.no_grad():
        output: Seq2SeqLMOutput = model.forward(
            input_token_ids=input_tokenized['input_ids'].to(device),
            input_att_mask=input_tokenized['attention_mask'].to(device),
            decoder_input_token_ids=decoder_input_toks.to(device),
            decoder_input_att_mask=decoder_input_att_mask.to(device),
            image_features=decoder_image_features.to(device),
            output_token_ids=output_toks.to(device)
        )
    logits: Tensor = output.logits
    loss_fn = nn.CrossEntropyLoss().to(device)
    losses: Tensor = torch.zeros(n_candidates)
    for i in range(n_candidates):
        target: Tensor = output_toks[i].to(device)
        loss = loss_fn(logits[i], target)
        losses[i] = loss
    selected_candidate = losses.argmin().item()
    accuracy = float((selected_candidate == action.target_object.index))
    mrr: float = reciprocal_rank(losses, action.target_object.index)
    if log_file is not None:
        print(f"{str(action): <40} acc={accuracy: .2f} mrr={mrr: .2f} \t {candidate_output_texts[selected_candidate]}",
              file=log_file)
    return accuracy, mrr


def evaluate_object_selection_by_action_type(model: ImageConditionedLLMOnDecoder,
                                            dataset: EvalAlfredHLActionDataset,
                                            scorer: Callable,
                                            log_file: Optional[TextIO] = None) -> Tuple[dict, dict]:
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

    accuracies = {}
    mrrs = {}
    for action_type in selected_action_types:
        print(f"Evaluating action {action_type}")
        actions = actions_by_type[action_type]
        accuracy, mrr = compute_object_accuracy_for_same_action_type(model, actions, scorer, log_file)
        accuracies[action_type] = accuracy
        mrrs[action_type] = mrr
    return accuracies, mrrs


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
                accuracy, mrr = scorer(subst_action, model)
                accuracies[subst_type].append(accuracy)
                mrrs[subst_type].append(mrr)
            except UnaccomplishedSubstitutionException as e:
                print(f"Skipping example: {e}")
    for key in subst_types:
        accuracies[key] = np.array(accuracies[key]).mean()
        mrrs[key] = np.array(mrrs[key]).mean()
    return accuracies, mrrs


def baseline_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir):
    with open(model_dir / "baseline_evaluation_by_action_type.log", 'w') as log_file:
        print(f"=== TRAIN ", "=" * 70, file=log_file)
        train_acc, train_mrr = evaluate_object_selection_by_action_type(
            model,
            train_dataset,
            compute_forced_metrics_for_action,
            log_file
        )
        print(f"\n\n=== VALID_SEEN ", "=" * 65, file=log_file)
        valid_seen_acc, valid_seen_mrr = evaluate_object_selection_by_action_type(
            model,
            valid_seen_dataset,
            compute_forced_metrics_for_action,
            log_file
        )
        print(f"\n\n=== VALID_UNSEEN ", "=" * 65, file=log_file)
        valid_unseen_acc, valid_unseen_mrr = evaluate_object_selection_by_action_type(
            model,
            valid_unseen_dataset,
            compute_forced_metrics_for_action,
            log_file
        )
    results_df = pd.DataFrame({
        'train_accuracy': train_acc,
        'valid_seen_accuracy': valid_seen_acc,
        'valid_unseen_accuracy': valid_unseen_acc,
        'train_mrr': train_mrr,
        'valid_seen_mrr': valid_seen_mrr,
        'valid_unseen_mrr': valid_unseen_mrr,
    })
    results_df.to_csv(model_dir / "baseline_evaluation_by_action_type.csv")


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
    train_acc = evaluate_object_substitution_by_object(model, train_forks, substitutions,
                                                       compute_forced_metrics_for_action)
    valid_seen_acc = evaluate_object_substitution_by_object(model, valid_seen_forks, substitutions,
                                                            compute_forced_metrics_for_action)
    valid_unseen_acc = evaluate_object_substitution_by_object(model, valid_unseen_forks, substitutions,
                                                              compute_forced_metrics_for_action)
    result_df = pd.DataFrame({
        'train_acc': train_acc,
        'valid_seen_acc': valid_seen_acc,
        'valid_unseen_acc': valid_unseen_acc,
        'train_mrr': train_acc,
        'valid_seen_mrr': valid_seen_acc,
        'valid_unseen_mrr': valid_unseen_acc
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

    baseline_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir)

    fork_substitution_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir)