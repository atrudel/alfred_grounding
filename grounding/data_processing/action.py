from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, List

import numpy as np
import skimage
from matplotlib import pyplot as plt
from torch import Tensor
from transformers import BatchEncoding

from grounding.data_processing.object import Object, bind_object, object_names
from grounding.models.base_models.clip import CLIPModelFrozen
from grounding.models.base_models.gpt2 import GPT2Model


class Action:
    def __init__(self, instruction: str, pddl: dict, image_resnet_features: Tensor, img_path: Path,
                 clip_model: CLIPModelFrozen, gpt_model, trajectory_path: Path, repeat_idx: int, split: str):
        self.id: Optional[int] = None
        self.pddl: dict = pddl
        self.trajectory_path: Path = trajectory_path
        self.repeat_idx: int = repeat_idx
        self.split: str = split
        self.type = pddl['discrete_action']['action']
        self.args = pddl['discrete_action']['args']
        self.target_object: Object = bind_object(self.args[0])

        # Instruction
        self.instruction: str = instruction
        self.instruction_clip_features: Tensor = self._extract_text_clip_features(instruction, clip_model)
        self.instruction_gpt_encoded: BatchEncoding = self._encode_text_for_gpt(instruction, gpt_model)

        # Image
        self.image_path: Path = img_path
        self.image: Optional[np.ndarray] = None
        self.image_resnet_features: Tensor = image_resnet_features
        self.image_clip_features: Tensor = self._extract_image_clip_features(img_path, clip_model)

        # Command
        self.templated_command: str = self._make_templated_string(self.type, self.args)
        self.command_clip_features: Tensor = self._extract_text_clip_features(self.templated_command, clip_model)
        self.command_gpt_encoded: BatchEncoding = self._encode_text_for_gpt(self.templated_command, gpt_model)

    def _extract_image_clip_features(self, image_path: Path, clip_model: CLIPModelFrozen) -> Tensor:
        raw_image: np.ndarray = self.load_image(image_path)
        return clip_model.encode_images(raw_image)

    @staticmethod
    def _extract_text_clip_features(text: str, clip_model: CLIPModelFrozen) -> Tensor:
        return clip_model.encode_texts(text)

    @staticmethod
    def _encode_text_for_gpt(text: str, gpt_model: GPT2Model) -> BatchEncoding:
        return gpt_model.tokenizer(text, return_tensors='pt')

    @staticmethod
    def _make_templated_string(action_type: str, args: List[str]) -> str:
        templated_str = ""

        if 'GotoLocation' in action_type:
            templated_str = "Go to the %s. " % (args[0])
        elif 'OpenObject' in action_type:
            templated_str = "Open the %s. " % (args[0])
        elif 'CloseObject' in action_type:
            templated_str = "Close the %s. " % (args[0])
        elif 'PickupObject' in action_type:
            templated_str = "Pick up the %s. " % (args[0])
        elif 'PutObject' in action_type:
            templated_str = "Put the %s in the %s. " % (args[0], args[1])
        elif 'CleanObject' in action_type:
            templated_str = "Wash the %s. " % (args[0])
        elif 'HeatObject' in action_type:
            templated_str = "Heat the %s. " % (args[0])
        elif 'CoolObject' in action_type:
            templated_str = "Cool the %s. " % (args[0])
        elif 'ToggleObject' in action_type:
            templated_str = "Toggle %s. " % (args[0])
        elif 'SliceObject' in action_type:
            templated_str = "Slice the %s. " % (args[0])
        elif 'NoOp' in action_type:
            templated_str = "<<STOP>>"
        return templated_str

    def make_command_options_for_all_objects(self):
        # Todo: Handle receptacle objects too
        return [self._make_templated_string(self.type, [object_name] + self.args[1:])
                for object_name in object_names]

    def make_substitution(self, substitution_word: str) -> Action:
        new_action: Action = copy.copy(self)
        new_action.instruction = self.instruction.replace(
            self.target_object.templated_string_form,
            substitution_word
        )
        if new_action.instruction == self.instruction:
            raise UnaccomplishedSubstitutionException(
                f"No occurrences of {self.target_object.templated_string_form} in {self.instruction}"
            )
        return new_action

    def assign_id(self, id: int) -> Action:
        self.id = id
        return self

    def __str__(self) -> str:
        split_abbreviations = {
            'train': 'TRAIN',
            'valid_seen': 'VSEEN',
            'valid_unseen': 'VUNSEEN'
        }
        return f"Act_{split_abbreviations[self.split]}_{self.id:0>6}_{self.type}({self.args})"

    def __repr__(self) -> str:
        return self.__str__()

    def show(self, image: bool = True) -> None:
        """Display function for notebooks"""
        print(self)
        print(self.trajectory_path)
        print(f"INSTRUCTION: {self.instruction}")
        print(f"COMMAND:     {self.templated_command}")
        if image:
            try:
                img: np.ndarray = skimage.io.imread(self.image_path)
                plt.figure(figsize=(5,5))
                plt.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
                skimage.io.imshow(img)
            except FileNotFoundError:
                print("File image not available: ", self.image_path)

    def get_image(self) -> np.ndarray:
        if self.image is None:
            self.load_and_store_image()
        return self.image

    def load_and_store_image(self) -> Action:
        self.image = self.load_image(self.image_path)
        return self

    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        return skimage.io.imread(image_path)


class UnaccomplishedSubstitutionException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"UnaccomplishedSubstitutionException: {self.message}"
