import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from config import REPO_ROOT
from grounding.data_processing.action import Action
from grounding.data_processing.trajectory import Trajectory, InconsistentTrajectoryException
from models.base_models.clip import CLIPModelFrozen

PREPROCESSED_ACTIONS_DIR_NAME: str = "preprocessed_actions"
DATASET_ROOT: Path = "alfred/data/json_feat_2.1.0"


def preprocess_all_data(skip_train: bool = False):
    with open('alfred/data/splits/oct21.json') as f:
        trajectories_by_splits = json.load(f)
    if skip_train:
        splits = ['valid_seen', 'valid_unseen']
    else:
        splits = ['train', 'valid_seen', 'valid_unseen']
    # Initialize CLIP model to preprocess images and text
    clip_model = CLIPModelFrozen()
    for split in splits:
        preprocess_split_in_high_level_actions(DATASET_ROOT, split,
                                               trajectories_by_splits, clip_model)


def preprocess_split_in_high_level_actions(dataset_root: str, split: str, trajectories_by_splits: Dict[str, List[dict]],
                                           clip_model: CLIPModelFrozen):
    split_root_path: Path = Path(dataset_root) / split
    dump_dir: Path = split_root_path / PREPROCESSED_ACTIONS_DIR_NAME
    os.makedirs(dump_dir, exist_ok=True)
    trajectory_metadata: List[dict] = trajectories_by_splits[split]

    print(f"Preprocessing {split} split in {split_root_path}")
    inconsistencies = 0
    count = 0
    for traj in tqdm(trajectory_metadata):
        traj_dir: Path = split_root_path / traj['task']
        trajectory: Trajectory = Trajectory(traj_dir, traj['repeat_idx'], split)
        try:
            actions: List[Action] = trajectory.split_actions(clip_model)
            for action in actions:
                action.assign_id(count)
                count += 1
                with open(dump_dir / f"{str(action)}.pickle", 'wb') as f:
                    pickle.dump(action, f)
        except InconsistentTrajectoryException:
            inconsistencies += 1

    print(f"Preprocessed actions saved in {dump_dir}")
    print(f"{inconsistencies}/{len(trajectory_metadata)} inconsistent trajectories thrown away.")



parser = argparse.ArgumentParser(description="Data Preprocessing")
parser.add_argument("--skip_train", action="store_true", help="Do not preprocess the training set.")


if __name__ == '__main__':
    assert Path(os.getcwd()).stem == 'alfred_grounding', ("The preprocessing script should be launched from the root "
                                                          "of the directory")
    args = parser.parse_args()
    preprocess_all_data(args.skip_train)