import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import pandas as pd
import torch

from grounding.data_processing.datasets_eval import EvalAlfredHLActionDataset


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


def load_eval_data(args):
    datasets: List[Tuple[str, EvalAlfredHLActionDataset]] = []
    if args.train is True:
        datasets += [('train', EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/train'))]
    datasets += [
        ('valid_seen', EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_seen')),
        ('valid_unseen', EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_unseen'))
    ]
    if args.debug:
        datasets = [(split, dataset[:10]) for split, dataset in datasets]
    return datasets


def get_checkpoint_path(args) -> Optional[Path]:
    checkpoint_dir: Path = Path(args.model_dir) / 'checkpoints'
    filenames: List[str] = os.listdir(checkpoint_dir)
    for filename in filenames:
        if filename.endswith('.ckpt'):
            return checkpoint_dir / filename
    return None


def apply_scoring_function_to_datasets(scoring_function: callable,
                                       datasets: List[Tuple[str, EvalAlfredHLActionDataset]],
                                       *args, **kwargs) -> pd.DataFrame:
    """Applies the scoring function to each dataset.
    args and kwargs must contain the relevant models and will be passed to the scoring function"""
    all_metrics: Dict[dict] = {}
    for split, dataset in datasets:
        print(f"Processing dataset: {split}..\n")
        metrics: dict = scoring_function(dataset, *args, **kwargs)
        all_metrics[split] = metrics
    return pd.DataFrame(all_metrics)


class Metrics(defaultdict):
    def __init__(self):
        super().__init__(list)

    def add(self, metric: str, value: float):
        self[metric].append(value)

    def add_many(self, metrics: Dict[str, List[float]]):
        for metric, values in metrics.items():
            if isinstance(values, list):
                self[metric].extend(values)
            elif isinstance(values, float):
                self[metric].append(values)
            elif isinstance(values, int):
                self[metric].append(values)
            else:
                raise ValueError(f"Unrecognized data type to add to {metric} in Metrics: {values}")

    def summarize(self) -> dict:
        return {key: np.array(values).mean() for key, values in self.items()}


        