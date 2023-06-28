import json
from pathlib import Path
from typing import List

import torch
from torch import Tensor

from grounding.data_processing.action import Action
from grounding.data_processing.object import Object

class Trajectory:
    def __init__(self,
                 directory: str,
                 repeat_idx: int,
                 ):
        self.directory: Path = Path(directory)
        self.repeat_idx: int = repeat_idx

        with open(self.directory / 'traj_data.json') as f:
            traj_data: dict = json.load(f)

        all_image_features = torch.load(self.directory / 'feat_conv.pt')

        self.task_type: str = traj_data['task_type']
        self.annotation: dict = traj_data['turk_annotations']['anns'][self.repeat_idx]
        self.pddl_plan: dict = self.fix_missing_end_action(traj_data['plan'])

        self.target_object: Object = Object(traj_data['pddl_params']['object_target'])
        self.receptacle_object: Object = Object(traj_data['pddl_params']['parent_target'])
        self.toggle_object: Object = Object(traj_data['pddl_params']['toggle_target'])

        self.image_features: List[Tensor] = self.extract_high_level_image_features(all_image_features, self.pddl_plan)

    @staticmethod
    def fix_missing_end_action(pddl_plan: dict) -> dict:
        # Append a terminal action to a sequence of high-level actions if it is missing
        high_level_pddl = pddl_plan['high_pddl']
        if high_level_pddl[-1]['planner_action']['action'] != 'End':
            high_level_pddl.append({
                'discrete_action': {'action': 'NoOp', 'args': []},
                'planner_action': {'value': 1, 'action': 'End'},
                'high_idx': len(high_level_pddl)
            })
        pddl_plan['high_pddl'] = high_level_pddl

        # Append a terminal action to the sequence of low-level actions
        end_low_action = {
            'api_action': {'action': 'NoOp'},
            'discrete_action': {'action': '<<stop>>', 'args': {}},
            'high_idx': high_level_pddl[-1]['high_idx']
        }
        pddl_plan['low_actions'].append(end_low_action)

        # Data integrity check
        if pddl_plan['high_pddl'][-1]['high_idx'] != pddl_plan['low_actions'][-1]['high_idx']:
            raise InconsistentTrajectoryException(f"Inconsistent high_idx between (fixed) high and low-level pddl")
        return pddl_plan

    @staticmethod
    def extract_high_level_image_features(all_image_features: Tensor,
                                          pddl_plan: dict
                                          ) -> List[Tensor]:
        """
        Visual features are provided for every low-level action. Here we want them for every high-level action.
        We extract the visual features associated with the first low-level action of each high-level action.
        We perform average pooling in order to get a 512-dim feature vector for each image.
        """
        num_high_level_actions: int = len(pddl_plan['high_pddl'])
        high_level_image_features: List[Tensor] = []

        # Extract the visual feature tensor of the first low-level action of each high-level action.
        for high_level_idx in range(num_high_level_actions):
            for low_level_idx, low_level_action in enumerate(pddl_plan['low_actions']):
                if low_level_action['high_idx'] == high_level_idx:
                    resnet_feats_3d: Tensor = all_image_features[low_level_idx,...]  # (512, 7, 7)
                    resnet_feats_1d: Tensor = torch.mean(torch.mean(resnet_feats_3d, 2), 1)  # (512)
                    high_level_image_features.append(resnet_feats_1d.unsqueeze(0))  # (1, 512)
                    break
        return high_level_image_features

    def split_actions(self) -> List[Action]:
        """Splits a trajectory into actions. Gets rid of the Stop action."""
        instructions: List[str] = self.annotation['high_descs']
        high_pddl_actions: List[dict] = self.pddl_plan['high_pddl'][:-1]
        image_features: List[Tensor] = self.image_features[:-1]

        if not len(instructions) == len(high_pddl_actions) == len(image_features):
            raise InconsistentTrajectoryException("Trajectory doesn't have equal number of instructions, pddl actions and image features to split into actions.")

        return [Action(instr, pddl, img)
                for instr, pddl, img in zip(instructions, high_pddl_actions, image_features)]

    def __str__(self) -> str:
        return f"Traj< {self.task_type}({self.target_object}, {self.receptacle_object}, {self.toggle_object}) >"

    def __len__(self) -> int:
        return len(self.pddl_plan)


class InconsistentTrajectoryException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"DataInconsistencyException: {self.message}"
