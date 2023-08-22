import os
import pickle
from pathlib import Path
from typing import List, Optional

from data_processing.action import Action
from data_processing.preprocess_data import DATASET_ROOT, PREPROCESSED_ACTIONS_DIR_NAME


def load_preprocessed_action(split: str, id: int) -> Optional[Action]:
    actions_directory: Path = DATASET_ROOT / split / PREPROCESSED_ACTIONS_DIR_NAME
    action_filenames: List[str] = os.listdir(actions_directory)
    for filename in action_filenames:
        if f"_{id}_" in filename:
            with open(actions_directory / filename, 'rb') as f:
                action: Action = pickle.load(f)
                return action
    return None


if __name__ == '__main__':
    action = load_preprocessed_action('train', 0)
    a = 1