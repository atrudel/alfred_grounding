"""
Evaluation of the alignment subtask for CLASP
"""
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from config import REPO_ROOT, DEVICE
from data_processing.action import Action
from data_processing.datasets_eval import EvalAlfredHLActionDataset
from evaluation.utils import parse_eval_args, announce_start_evaluation, get_checkpoint_path, load_eval_data
from models.base_models.clip import CLIPModelFrozen
from models.clasp import CLASP


def calculate_alignment_metrics(clasp_model: CLASP, clip_model: CLIPModelFrozen, dataset: EvalAlfredHLActionDataset):
    testable_actions: List[Action] = dataset.get_testable_actions()
    accuracies = []
    mrrs = []
    for action in tqdm(testable_actions):
        command_options: List[str] = action.make_command_options_for_all_objects()
        commands_clip_features = clip_model.encode_texts(command_options)
        z_instruction: Tensor = clasp_model._encode_instructions(action.instruction_clip_features)
        z_behaviors: Tensor = clasp_model._encode_behaviors(
            action.image_clip_features.repeat(commands_clip_features.shape[0], 1),
            commands_clip_features
        )
        ground_truth: int = action.get_object_index()
        accuracy, mrr = _comparative_z_metrics(z_instruction, z_behaviors, ground_truth)
        accuracies.append(accuracy)
        mrrs.append(mrr)

    return {
        'accuracy': np.array(accuracies).mean(),
        'mrr': np.array(mrrs).mean()
    }


def _comparative_z_metrics(z_instruction: Tensor, z_behavior_versions: Tensor, ground_truth_index: int) -> Tuple[float, float]:
    product: Tensor = torch.matmul(z_behavior_versions, z_instruction)
    accuracy: float = float(product.argmax(dim=0) == ground_truth_index)

    _, indices = torch.sort(product, descending=True)
    rank: int = torch.where(indices == ground_truth_index)[0].item() + 1
    mrr: float = 1. / rank
    return accuracy, mrr


def apply_metrics_calculation_to_datasets(metric_function: callable,
                                          clasp_model: CLASP,
                                          clip_model: CLIPModelFrozen,
                                          datasets: List[Tuple[str, EvalAlfredHLActionDataset]])-> pd.DataFrame:
    all_metrics: Dict[dict] = {}
    for split, dataset in datasets:
        print(f"Processing dataset {split}")
        metrics = metric_function(clasp_model, clip_model, dataset)
        all_metrics[split] = metrics
    return pd.DataFrame(all_metrics)


if __name__ == '__main__':
    description = "Alignment evaluation"
    args: Namespace = parse_eval_args(description)
    announce_start_evaluation(args)

    print("Loading data...")
    datasets: List[Tuple[str, EvalAlfredHLActionDataset]] = load_eval_data(args)

    checkpoint_path: Path = get_checkpoint_path(args)
    print(f"Loading model from {checkpoint_path}...")
    clasp_model: CLASP = CLASP.load_from_checkpoint(checkpoint_path, map_location=DEVICE)

    print("Instantiating CLIP model...")
    clip_model: CLIPModelFrozen = CLIPModelFrozen()

    print("Performing alignment evaluation...")
    results = apply_metrics_calculation_to_datasets(
        metric_function=calculate_alignment_metrics,
        clasp_model=clasp_model,
        clip_model=clip_model,
        datasets=datasets
    )
    print(results)
    save_file: Path = args.model_dir / f"alignment_evaluation.csv"
    results.to_csv(save_file)
    print(f"Results_saved to {save_file}")
