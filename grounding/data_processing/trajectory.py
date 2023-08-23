import json
from pathlib import Path
from typing import List

import torch
from torch import Tensor

from grounding.data_processing.action import Action
from grounding.data_processing.object import Object, bind_object, UnmatchedObjectException
from grounding.models.base_models.clip import CLIPModelFrozen
from models.base_models.gpt2 import PrefixGPT2Model


class Trajectory:
    def __init__(self, directory: str, repeat_idx: int, split: str):
        self.directory: Path = Path(directory)
        self.repeat_idx: int = repeat_idx
        self.split = split

        with open(self.directory / 'traj_data.json') as f:
            traj_data: dict = json.load(f)

        all_image_features = torch.load(self.directory / 'feat_conv.pt')
        all_image_info: List[dict] = traj_data['images']

        self.task_type: str = traj_data['task_type']
        self.annotation: dict = traj_data['turk_annotations']['anns'][self.repeat_idx]
        self.pddl_plan: dict = self.fix_missing_end_action(traj_data['plan'])

        self.target_object: Object = bind_object(traj_data['pddl_params']['object_target'])
        # self.receptacle_object: Object = bind_object(traj_data['pddl_params']['parent_target'])
        # self.toggle_object: Object = bind_object(traj_data['pddl_params']['toggle_target'])

        self.image_features: List[Tensor] = self.extract_high_level_image_features(all_image_features, self.pddl_plan)
        self.image_paths: List[Path] = self.extract_high_level_image_paths(all_image_info, self.directory)
        a = 1

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

    @staticmethod
    def extract_high_level_image_paths(all_image_info: List[dict], directory: Path) -> List[Path]:
        """Extract the full path of the image that corresponds to the beginning of each high-level action"""
        def get_full_image_path(directory: Path, image_filename: str) -> Path:
            return Path('alfred/data/full_2.1.0') / \
                    directory.relative_to('alfred/data/json_feat_2.1.0') / \
                    'raw_images' / \
                    image_filename.replace('png', 'jpg')

        image_paths: List[Path] = []
        curr_high_lvl_idx = -1
        # Add first image of each high-level action
        for image_info in all_image_info:
            if image_info['high_idx'] > curr_high_lvl_idx:
                image_filename: str = image_info['image_name']
                image_path: Path = get_full_image_path(directory, image_filename)
                image_paths.append(image_path)
                curr_high_lvl_idx: int = image_info['high_idx']
        return image_paths

    def split_actions(self, clip_model: CLIPModelFrozen, gpt_model: PrefixGPT2Model) -> List[Action]:
        """
        Splits a trajectory into actions. Gets rid of the Stop action.
        Perform the pre-processing relative to the various models used in the project.
        """
        instructions: List[str] = self.annotation['high_descs']
        high_pddl_actions: List[dict] = self.pddl_plan['high_pddl'][:-1]
        image_features: List[Tensor] = self.image_features[:-1]
        image_paths: List[Path] = self.image_paths

        if not len(instructions) == len(high_pddl_actions) == len(image_features):
            raise InconsistentTrajectoryException("Trajectory doesn't have equal number of instructions, pddl actions and image features to split into actions.")

        actions = []
        for instr, pddl, img_feats, img_path in zip(instructions, high_pddl_actions, image_features, image_paths):
            try:
                actions.append(
                    Action(instr, pddl, img_feats, img_path, clip_model, gpt_model, self.directory, self.repeat_idx,
                           self.split))
            except UnmatchedObjectException:
                print(f"Action with unknown object rejected: {pddl['discrete_action']['action']}({pddl['discrete_action']['args']})")
        return actions

    def __str__(self) -> str:
        return f"Traj< {self.task_type}({self.target_object}) >"   #, {self.receptacle_object}, {self.toggle_object}) >"

    def __len__(self) -> int:
        return len(self.pddl_plan)


class InconsistentTrajectoryException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"DataInconsistencyException: {self.message}"
