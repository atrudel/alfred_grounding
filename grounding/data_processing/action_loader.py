import os
import pickle
from pathlib import Path
from typing import List, Optional

from grounding.data_processing.action import Action
from grounding.data_processing.preprocess_data import DATASET_ROOT, PREPROCESSED_ACTIONS_DIR_NAME


def load_preprocessed_action(split: str, id: int) -> Optional[Action]:
    actions_directory: Path = DATASET_ROOT / split / PREPROCESSED_ACTIONS_DIR_NAME
    action_filenames: List[str] = sorted(os.listdir(actions_directory))
    filename = action_filenames[id]
    with open(actions_directory / filename, 'rb') as f:
        action: Action = pickle.load(f)
    return action
