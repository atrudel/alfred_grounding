import argparse
from pathlib import Path
from typing import Tuple, TextIO, Optional

import pandas as pd
import torch
from torch import Tensor
from transformers import BatchEncoding

from grounding.data_processing.action import Action
from grounding.data_processing.datasets import EvalAlfredHLActionDataset
from grounding.evaluation.evaluate_object_selection import evaluate_object_selection_by_action_type, \
    evaluate_object_substitution_by_object
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Evaluation on object selection.')
parser.add_argument('--model_dir', type=str, help='Directory where the model checkpoint is located.')
parser.add_argument('--debug', action='store_true')


def compute_accuracy(action: Action, output_text: str) -> float:
    return float(action.target_object.templated_string_form in output_text)


def compute_mrr(action: Action, model, object_logits: Tensor) -> float:
    target_object_tokens = model.tokenizer(action.target_object.templated_string_form)
    target_object_first_tok: int = target_object_tokens['input_ids'][0]

    _, indices = torch.sort(object_logits, descending=True)
    rank = torch.where(indices.squeeze() == target_object_first_tok)[0].item() + 1
    return 1. / rank


def compute_generative_metrics_for_action(action: Action,
                                          model: ImageConditionedLLMOnDecoder,
                                          log_file: Optional[TextIO]) -> Tuple[float, float]:
    input_tokenized: BatchEncoding = model.tokenizer(action.instruction, return_tensors='pt')

    # Decoder input (e.g. "Pick up the")
    decoder_input_text: str = action.templated_string.replace(f"{action.target_object.templated_string_form}. ", '')
    decoder_input_tokenized: BatchEncoding = model.tokenizer(decoder_input_text, return_tensors='pt')
    decoder_input_toks: Tensor = decoder_input_tokenized['input_ids'][:, :-1]
    decoder_input_att_mask: Tensor = decoder_input_tokenized['attention_mask'][:, :-1]

    image_features: Tensor = action.image_features.unsqueeze(0)

    output_texts, object_logits = model.generate(
        input_token_ids=input_tokenized['input_ids'],
        input_att_mask=input_tokenized['attention_mask'],
        decoder_input_token_ids=decoder_input_toks,
        decoder_input_att_mask=decoder_input_att_mask,
        image_features=image_features.to(device)
    )
    accuracy: float = compute_accuracy(action, output_texts[0])
    mrr: float = compute_mrr(action, model, object_logits)
    if log_file is not None:
        print(f"{str(action): <40} acc={accuracy: .2f} mrr={mrr: .2f} \t {output_texts[0]}", file=log_file)
    return accuracy, mrr


def generative_baseline_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir):
    with open(model_dir / "generative_baseline_evaluation_by_action_type.log", 'w') as log_file:
        print(f"=== TRAIN ", "=" * 70, file=log_file)
  #      train_acc, train_mrr = evaluate_object_selection_by_action_type(
  #          model,
  #          train_dataset,
  #          compute_generative_metrics_for_action,
  #          log_file
  #      )
        print(f"\n\n=== VALID_SEEN ", "=" * 65, file=log_file)
        valid_seen_acc, valid_seen_mrr = evaluate_object_selection_by_action_type(
            model,
            valid_seen_dataset,
            compute_generative_metrics_for_action,
            log_file
        )
        print(f"\n\n=== VALID_UNSEEN ", "=" * 65, file=log_file)
        valid_unseen_acc, valid_unseen_mrr = evaluate_object_selection_by_action_type(
            model,
            valid_unseen_dataset,
            compute_generative_metrics_for_action,
            log_file
        )
    results_df = pd.DataFrame({
#        'train_accuracy': train_acc,
        'valid_seen_accuracy': valid_seen_acc,
        'valid_unseen_accuracy': valid_unseen_acc,
 #       'train_mrr': train_mrr,
        'valid_seen_mrr': valid_seen_mrr,
        'valid_unseen_mrr': valid_unseen_mrr,
    })
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
                                                            compute_generative_metrics_for_action)
    valid_unseen_acc, valid_unseen_mrr = evaluate_object_substitution_by_object(model, valid_unseen_forks, substitutions,
                                                              compute_generative_metrics_for_action)

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
    args: argparse.Namespace = parser.parse_args()
    model_dir: Path = Path(args.model_dir)
    model_path = model_dir / "checkpoint.pth.tar"

    model = ImageConditionedLLMOnDecoder.load(model_path).to(device)

#    train_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/train')
    train_dataset = None
    valid_seen_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_seen')
    valid_unseen_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_unseen')

    generative_baseline_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir)
    generative_fork_substitution_evaluation(model, train_dataset, valid_seen_dataset, valid_unseen_dataset, model_dir)
