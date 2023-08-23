from collections import defaultdict
from typing import Optional, List, Dict

from grounding.data_processing.action import Action
from grounding.data_processing.datasets_train import AlfredHLActionDataset


class EvalAlfredHLActionDataset(AlfredHLActionDataset):
    """Dataset of high level actions used to evaluate models in grounding."""
    debug_max_actions = 5

    def __init__(self, root_dir: str):
        super().__init__(root_dir=root_dir)

    def get_actions_by_objects(self, action_types: Optional[List[str]] = None, no_repeat: bool = True) -> Dict[str, List[Action]]:
        actions_by_objects: defaultdict = defaultdict(list)
        for action_idx in range(len(self)):
            action: Action = self.get_action(action_idx)
            if no_repeat and action.repeat_idx != 0:  # no_repeat == Keep only one annotation per action
                continue
            if action_types and action.type not in action_types:
                continue
            if self.debug and len(actions_by_objects[action.target_object.name]) >= self.debug_max_actions:
                continue
            actions_by_objects[action.target_object.name].append(action)
        return actions_by_objects

    def get_actions_by_type(self) -> Dict[str, List[Action]]:
        actions_by_type: defaultdict = defaultdict(list)
        for action_idx in range(len(self)):
            action: Action = self.get_action(action_idx)
            if self.debug and len(actions_by_type[action.type]) >= self.debug_max_actions:
                continue
            actions_by_type[action.type].append(action)
        return actions_by_type

    def get_actions_by_indices(self, indices: List[int]) -> List[Action]:
        return [self.get_action(index) for index in indices]

    def inspect_action(self, index: int) -> None:
        self.get_action(index).show()


