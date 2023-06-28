from __future__ import annotations

from typing import Optional

from torch import Tensor

from grounding.data_processing.object import Object


class Action:
    def __init__(self, instruction: str, pddl: dict, image_features: Tensor):
        self.id: Optional[int] = None
        self.instruction: str = instruction
        self.pddl: dict = pddl
        self.image_features: Tensor = image_features

        self.type = pddl['discrete_action']['action']
        self.args = pddl['discrete_action']['args']
        self.target_object: Object = Object(self.args[0])
        self.templated_string: str = self.convert_pddl_to_templated_string(pddl)

    @staticmethod
    def convert_pddl_to_templated_string(pddl_action: dict) -> str:
        templated_str = ""
        a_type = pddl_action['discrete_action']['action']
        args = pddl_action['discrete_action']['args']

        if 'GotoLocation' in a_type:
            templated_str = "Go to the %s. " % (args[0])
        elif 'OpenObject' in a_type:
            templated_str = "Open the %s. " % (args[0])
        elif 'CloseObject' in a_type:
            templated_str = "Close the %s. " % (args[0])
        elif 'PickupObject' in a_type:
            templated_str = "Pick up the %s. " % (args[0])
        elif 'PutObject' in a_type:
            templated_str = "Put the %s in the %s. " % (args[0], args[1])
        elif 'CleanObject' in a_type:
            templated_str = "Wash the %s. " % (args[0])
        elif 'HeatObject' in a_type:
            templated_str = "Heat the %s. " % (args[0])
        elif 'CoolObject' in a_type:
            templated_str = "Cool the %s. " % (args[0])
        elif 'ToggleObject' in a_type:
            templated_str = "Toggle %s. " % (args[0])
        elif 'SliceObject' in a_type:
            templated_str = "Slice the %s. " % (args[0])
        elif 'NoOp' in a_type:
            templated_str = "<<STOP>>"
        return templated_str

    def assign_id(self, id: int) -> Action:
        self.id = id
        return self

    def __str__(self) -> str:
        return f"Act{self.id}_{self.type}({self.args})"

    def __repr__(self) -> str:
        return self.__str__()