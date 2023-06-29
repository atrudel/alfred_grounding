from __future__ import annotations

from typing import Optional, List

from torch import Tensor

from grounding.data_processing.object import Object, bind_object, object_names


class Action:
    def __init__(self, instruction: str, pddl: dict, image_features: Tensor):
        self.id: Optional[int] = None
        self.instruction: str = instruction
        self.pddl: dict = pddl
        self.image_features: Tensor = image_features

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


    def assign_id(self, id: int) -> Action:
        self.id = id
        return self

    def __str__(self) -> str:
        return f"Act{self.id}_{self.type}({self.args})"

    def __repr__(self) -> str:
        return self.__str__()