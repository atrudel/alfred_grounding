import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm
from transformers import BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput

from grounding.data_processing.action import Action
from grounding.data_processing.datasets import EvalAlfredHLActionDataset
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Evaluation on object selection.')
parser.add_argument('--model_dir', type=str, help='Directory where the model checkpoint is located.')



def compute_object_accuracy_for_same_action_type(model: ImageConditionedLLMOnDecoder, actions: List[Action]):
    accuracies = []
    for action in tqdm(actions):
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
            output: Seq2SeqLMOutput = model.train_forward(
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

        accuracy = float((losses.argmin().item() == action.target_object.index))
        accuracies.append(accuracy)

    return np.array(accuracies).mean()


def evaluate_object_selection_by_action_type(model: ImageConditionedLLMOnDecoder,
                                            dataset: EvalAlfredHLActionDataset) -> dict:
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

    results = {}
    for action_type in selected_action_types:
        print(f"Evaluating action {action_type}")
        actions = actions_by_type[action_type]
        accuracy = compute_object_accuracy_for_same_action_type(model, actions)
        results[action_type] = accuracy
    return results





if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args()
    model_dir: Path = Path(args.model_dir)
    model_path = model_dir / "checkpoint.pth.tar"

    model = ImageConditionedLLMOnDecoder.load(model_path).to(device)

    train_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/train')
    valid_seen_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_seen')
    valid_unseen_dataset = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_unseen')

    train_accuracies: dict = evaluate_object_selection_by_action_type(model, train_dataset)
    valid_seen_accuracies: dict = evaluate_object_selection_by_action_type(model, valid_seen_dataset)
    valid_unseen_accuracies: dict = evaluate_object_selection_by_action_type(model, valid_unseen_dataset)

    results_df = pd.DataFrame({
        'train_accuracy': train_accuracies,
        'valid_seen_accuracy': valid_seen_accuracies,
        'valid_unseen_accuracy': valid_unseen_accuracies,
    })
    results_df.to_csv(model_dir / "object_selection_by_action_type.csv")
