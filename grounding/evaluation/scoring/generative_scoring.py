from typing import Optional, TextIO, Dict

import torch
from torch import Tensor
from transformers import BatchEncoding

from grounding.data_processing.action import Action
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_accuracy(action: Action, output_text: str) -> float:
    return float(action.target_object.templated_string_form in output_text)


def compute_generative_accuracy_for_action(action: Action,
                                           model: ImageConditionedLLMOnDecoder,
                                           log_file: Optional[TextIO]) -> Dict[str,float]:

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
    if log_file is not None:
        print(f"{str(action): <40} acc={accuracy: .2f}\t {output_texts[0]}", file=log_file)
    return {'accuracy': accuracy}
