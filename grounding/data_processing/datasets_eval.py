from collections import defaultdict
from typing import Optional, List, Dict

from data_processing.action import Action
from data_processing.datasets_train import AlfredHLActionDataset


class EvalAlfredHLActionDataset(AlfredHLActionDataset):
    """Dataset of high level actions used to evaluate models in grounding."""
    debug_max_actions = 5

    def get_actions_by_objects(self, action_types: Optional[List[str]] = None, no_repeat: bool = True) -> Dict[str, List[Action]]:
        actions_by_objects: defaultdict = defaultdict(list)
        for action in self.actions:
            if no_repeat and action.repeat_idx != 0:
                continue
            if action_types and action.type not in action_types:
                continue
            if self.debug and len(actions_by_objects[action.target_object.name]) >= self.debug_max_actions:
                continue
            actions_by_objects[action.target_object.name].append(action)
        return actions_by_objects

    def get_actions_by_type(self) -> Dict[str, List[Action]]:
        actions_by_type: defaultdict = defaultdict(list)
        for action in self.actions:
            if self.debug and len(actions_by_type[action.type]) >= self.debug_max_actions:
                continue
            actions_by_type[action.type].append(action)
        return actions_by_type

    def get_actions_by_indices(self, indices: List[int]) -> List[Action]:
        return [self[index] for index in indices]

    def inspect_action(self, index: int) -> None:
        self.actions[index].show()

    def __getitem__(self, item: int) -> Action:
        return self.actions[item]
