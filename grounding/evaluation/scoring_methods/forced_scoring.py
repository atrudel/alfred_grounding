from typing import Optional, TextIO, Tuple, List, Dict

import torch
from torch import Tensor, nn as nn

from grounding.data_processing.action import Action
from grounding.data_processing.object import Object, object_names
from grounding.models.clasp import CLASP

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_forced_metrics_for_single_action(action: Action,
                                             model: nn.Module,
                                             log_file: Optional[TextIO] = None) -> Dict[str, float]:
    candidate_output_texts, losses = _compute_forced_losses(action, model)
    accuracy, selected_candidate = compute_accuracy(action, losses)
    mrr: float = reciprocal_rank(losses, action.target_object.index)
    if log_file is not None:
        print(f"{str(action): <40} acc={accuracy: .2f} mrr={mrr: .2f} \t {candidate_output_texts[selected_candidate]}",
              file=log_file)
    return {'accuracy': accuracy, 'mrr': mrr}


def compute_accuracy(action: Action, losses: Tensor):
    selected_candidate = losses.argmin().item()
    accuracy = float((selected_candidate == action.target_object.index))
    return accuracy, selected_candidate


def reciprocal_rank(losses: Tensor, target_index) -> float:
    _, indices = torch.sort(losses)
    rank = torch.where(indices == target_index)[0].item() + 1
    return 1. / rank

def reciprocal_rank_with_similarity(similarities: Tensor, target_index) -> float:
    _, indices = torch.sort(similarities, descending=True)
    rank = torch.where(indices == target_index)[0].item() + 1
    return 1. / rank

def compute_forced_metrics_for_ambiguous_situation(action: Action,
                                                   model: nn.Module,
                                                   tested_objects: List[Object]
                                                   ) -> Tuple[List[float], str]:
    _, losses = _compute_forced_losses(action, model)
    mrrs: List[float] = [reciprocal_rank(losses, object.index) for object in tested_objects]
    most_likely_index: int = losses.argmin().item()
    most_likely_object: str = object_names[most_likely_index]
    return mrrs, most_likely_object

def compute_alignment_metrics_for_ambiguous_situation(action: Action,
                                                      clasp_model: CLASP,
                                                      tested_objects: List[Object]) -> Tuple[List[float], str]:
    # Generate candidate commands
    candidate_output_texts: List[str] = action.make_command_options_for_all_objects()

    # Model scores them
    similarities: Tensor = clasp_model.evaluate_candidate_commands_through_alignment(action, candidate_output_texts)  # [80]

    # Calculate MRRS
    mrrs: List[float] = [reciprocal_rank_with_similarity(similarities, object.index) for object in tested_objects]

    # Extract most likely
    most_likely_object_idx: int = similarities.argmax().item()
    most_likely_object: str = object_names[most_likely_object_idx]

    return mrrs, most_likely_object



def _compute_forced_losses(action: Action, model: nn.Module):
    candidate_output_texts: List[str] = action.make_command_options_for_all_objects()
    logits, output_toks = model.evaluate_candidate_commands_on_all_object_options(
        action, candidate_output_texts
    )
    n_candidates = len(candidate_output_texts)
    loss_fn = nn.CrossEntropyLoss().to(device)
    losses: Tensor = torch.zeros(n_candidates)
    for i in range(n_candidates):
        target: Tensor = output_toks[i].to(device)
        loss = loss_fn(logits[i], target)
        losses[i] = loss
    return candidate_output_texts, losses

