import os
from pathlib import Path

GROUNDING_EVALUATION_FILENAME = 'grounding_evaluation'


def get_model_path(args):
    model_filename: str = os.listdir(args.model_dir / 'checkpoints')[-1]
    model_path: Path = args.model_dir / 'checkpoints' / model_filename
    return model_path
