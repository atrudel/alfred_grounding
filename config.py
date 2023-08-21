from datatest import working_directory
from pathlib import Path
import os

import torch


def absolute(path: str):
    with working_directory(__file__):
        absolute_path = os.path.abspath(path)
        return Path(absolute_path)


REPO_ROOT = absolute('.')
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
