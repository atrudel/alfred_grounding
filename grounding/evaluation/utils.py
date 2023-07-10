import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch


def parse_eval_args(description: str = "Model evaluation routine") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model checkpoint is located.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--train', action='store_true', help='Evaluate on training set also (takes longer).')

    args: argparse.Namespace = parser.parse_args()
    model_dir: Path = Path(args.model_dir)
    model_path: Path = model_dir / "checkpoint.pth.tar"
    args.model_dir = model_dir
    args.model_path = model_path

    args.description = description
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

def announce_start_evaluation(args: argparse.Namespace) -> None:
    print('##############################')
    print(args.description)
    print('Using train dataset: ', args.train)
    print('Device: ', args.device)
    print('##############################')
    print()


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
