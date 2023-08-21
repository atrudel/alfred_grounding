import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from grounding.data_processing.action import Action
from grounding.data_processing.trajectory import Trajectory, InconsistentTrajectoryException


high_level_action_dump_filename = 'high_level_actions.pickle'

parser = argparse.ArgumentParser(description="Data Preprocessing")
parser.add_argument("--skip_train", action="store_true", help="Do not preprocess the training set.")


def preprocess_all_data(skip_train: bool = False):
    with open('alfred/data/splits/oct21.json') as f:
        trajectories_by_splits = json.load(f)
    if skip_train:
        splits = ['valid_seen', 'valid_unseen']
    else:
        splits = ['train', 'valid_seen', 'valid_unseen']
    for split in splits:
        preprocess_split_in_high_level_actions('alfred/data/json_feat_2.1.0', split, trajectories_by_splits)


def preprocess_split_in_high_level_actions(dataset_root: str, split: str, trajectories_by_splits: Dict[str, List[dict]]):
    split_root_path: Path = Path(dataset_root) / split
    trajectory_ids: List[dict] = trajectories_by_splits[split]

    print(f"Preprocessing {split} split in {split_root_path}")
    preprocessed_actions = []
    inconsistencies = 0
    for traj in tqdm(trajectory_ids):
        traj_dir: Path = split_root_path / traj['task']
        trajectory: Trajectory = Trajectory(traj_dir, traj['repeat_idx'], split)
        try:
            actions: List[Action] = trajectory.split_actions()
            preprocessed_actions.extend(actions)
        except InconsistentTrajectoryException:
            inconsistencies += 1

    preprocessed_actions = [action.assign_id(id) for id, action in enumerate(preprocessed_actions)]

    dump_path = split_root_path / high_level_action_dump_filename
    with open(dump_path, 'wb') as f:
        pickle.dump(preprocessed_actions, f)
    print(f"Saved in {dump_path}")
    print(f"{inconsistencies}/{len(trajectory_ids)} inconsistent trajectories thrown away.")


if __name__ == '__main__':
    assert Path(os.getcwd()).stem == 'alfred_grounding', ("The preprocessing script should be launched from the root "
                                                          "of the directory")
    args = parser.parse_args()
    preprocess_all_data(args.skip_train)