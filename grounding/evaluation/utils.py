import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Model evaluation script.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model checkpoint is located.')
    parser.add_argument('--debug', action='store_true')

    args: argparse.Namespace = parser.parse_args()
    model_dir: Path = Path(args.model_dir)
    model_path: Path = model_dir / "checkpoint.pth.tar"
    args.model_dir = model_dir
    args.model_path = model_path
    return args


def mean(values: List[float], decimals: int = 3) -> float:
    return np.array(values).mean().round(decimals=decimals)


def object_counts(values: List[str]) -> str:
    """
    ['potato', 'lettuce', 'potato', 'potato', 'potato'] -> "potato(4), lettuce"
    """
    values = pd.Series(values)
    output = ""
    for obj, count in values.value_counts().items():
        output += obj
        if count > 1:
            output += f"({count})"
        output += ", "
    output = output[:-2]  # remove extra ", "
    return output
