from typing import Optional, TextIO, Tuple, List, Dict

import torch
from torch import Tensor, nn as nn
from transformers import BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput

from grounding.data_processing.action import Action
from grounding.data_processing.object import Object, object_names
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_forced_metrics_for_single_action(action: Action,
                                             model: ImageConditionedLLMOnDecoder,
                                             log_file: Optional[TextIO]) -> Dict[str, float]:
    candidate_output_texts, losses = compute_forced_losses(action, model)
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


def compute_forced_metrics_for_ambiguous_situation(action: Action,
                                                   model: ImageConditionedLLMOnDecoder,
                                                   tested_objects: List[Object]
                                                   ) -> Tuple[float, str]:
    _,  losses = compute_forced_losses(action, model)
    mrrs = [reciprocal_rank(losses, object.index) for object in tested_objects]
    most_likely_index: int = losses.argmin().item()
    most_likely_object: str = object_names[most_likely_index]
    return mrrs, most_likely_object


def compute_forced_losses(action: Action, model: ImageConditionedLLMOnDecoder):
    candidate_output_texts: List[str] = action.make_classification_strings()
    n_candidates = len(candidate_output_texts)
    input_tokenized: BatchEncoding = model.tokenizer(
        [action.instruction] * n_candidates,
        return_tensors='pt'
    )
    image_features: Tensor = action.image_features.unsqueeze(0).repeat(n_candidates, 1, 1)
    decoder_input_toks, decoder_input_att_mask, decoder_image_features, output_toks = model.prepare_decoder_input_output_data(
        image_features, candidate_output_texts)
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
    return candidate_output_texts, losses
