import json
import pickle
from pathlib import Path
from typing import List

from tqdm import tqdm

from grounding.data_processing.action import Action
from grounding.data_processing.trajectory import Trajectory, InconsistentTrajectoryException


high_level_action_dump_filename = 'high_level_actions.pickle'


def preprocess_all_data():
    with open('alfred/data/splits/oct21.json') as f:
        splits = json.load(f)
    train_split = splits['train']
    val_seen_split = splits['valid_seen']
    val_unseen_split = splits['valid_unseen']

    preprocess_split_in_high_level_actions('alfred/data/json_feat_2.1.0/train', train_split)
    preprocess_split_in_high_level_actions('alfred/data/json_feat_2.1.0/valid_seen', val_seen_split)
    preprocess_split_in_high_level_actions('alfred/data/json_feat_2.1.0/valid_unseen', val_unseen_split)



def preprocess_split_in_high_level_actions(path: str, trajectory_ids: List[dict]):
    print(f"Preprocessing split in {path}")
    split_root_path = Path(path)
    preprocessed_actions = []
    inconsistencies = 0
    for traj in tqdm(trajectory_ids):
        traj_dir: Path = split_root_path / traj['task']
        trajectory: Trajectory = Trajectory(traj_dir, traj['repeat_idx'])
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
    preprocess_all_data()