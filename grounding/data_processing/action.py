from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor

from grounding.data_processing.object import Object, bind_object, object_names
import skimage


class Action:
    def __init__(self, instruction: str, pddl: dict, image_features: Tensor, img_path: Path,
                 repeat_idx: int):
        self.id: Optional[int] = None
        self.instruction: str = instruction
        self.pddl: dict = pddl
        self.image_features: Tensor = image_features
        self.image_path: Path = img_path
        self.image: Optional[np.ndarray] = None
        self.repeat_idx: int = repeat_idx

        self.type = pddl['discrete_action']['action']
        self.args = pddl['discrete_action']['args']
        self.target_object: Object = bind_object(self.args[0])
        self.templated_string: str = self._make_templated_string(self.type, self.args)

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

    def make_classification_strings(self):
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
        return f"Act{self.id}_{self.type}({self.args})"

    def __repr__(self) -> str:
        return self.__str__()

    def show(self, image: bool = True) -> None:
        """Display function for notebooks"""
        print(self)
        print(f"INSTRUCTION: {self.instruction}")
        print(f"COMMAND:     {self.templated_string}")
        if image:
            try:
                img: np.ndarray = skimage.io.imread(self.image_path)
                plt.figure(figsize=(5,5))
                plt.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
                skimage.io.imshow(img)
            except FileNotFoundError:
                print("File image not available: ", self.image_path)

    def get_image(self) -> np.ndarray:
        return skimage.io.imread(self.image_path)

    def load_image(self) -> Action:
        self.image = self.get_image()
        return self

class UnaccomplishedSubstitutionException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"UnaccomplishedSubstitutionException: {self.message}"
