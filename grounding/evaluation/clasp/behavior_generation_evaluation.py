import os
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Dict, TextIO

import pandas as pd
from torch import nn
from tqdm import tqdm

from config import DEVICE
from data_processing.action import Action
from data_processing.datasets_eval import EvalAlfredHLActionDataset
from evaluation.scoring_methods.forced_scoring import compute_forced_metrics_for_single_action
from evaluation.scoring_methods.generative_scoring import compute_generative_accuracy_for_action
from evaluation.utils import parse_eval_args, announce_start_evaluation, load_eval_data, get_checkpoint_path, \
    apply_scoring_function_to_datasets, Metrics
from models.clasp import CLASP


def evaluate_object_forced_metrics_in_behavior_generation(dataset: EvalAlfredHLActionDataset,
                                                            clasp_model: CLASP,
                                                            log_file: TextIO) -> Dict:
    all_metrics: Metrics = Metrics()
    testable_actions: List[Action] = dataset.get_testable_actions()
    for action in tqdm(testable_actions):
        metrics: dict = compute_forced_metrics_for_single_action(action, clasp_model, log_file)
        all_metrics.add_many(metrics)
    return all_metrics.summarize()

def evaluate_object_generative_accuracy_in_behavior_generation(dataset: EvalAlfredHLActionDataset,
                                                                    clasp_model: CLASP):
    # accuracies: Metrics = Metrics()
    # testable_actions: List[Action] = dataset.get_testable_actions()
    # for action in tqdm(testable_actions):
    #     # TODO: verifier les methodes du modele appelees dans cette fonction
    #     # Todo: log
    #     accuracy: dict = compute_generative_accuracy_for_action(action, clasp_model)
    #     accuracies.add_many(accuracy)
    # return accuracies.summarize()
    pass

def evaluate_global_accuracy_and_qualitative_in_behavior_generation(dataset: EvalAlfredHLActionDataset,
                                                                    clasp_model: CLASP,
                                                                    ouput_file) -> Dict[str, float]:
    # accuracies = []
    # testable_actions: List[Action] = dataset.get_testable_actions()
    # for action in tqdm(testable_actions):
    #     predicted_action: str = clasp_model.generate_behavior(
    #         instruction_clip_features=action.instruction_clip_features,
    #         image_clip_features=action.image_clip_features
    #     )
    #     accuracy = _global_generation_accuracy(predicted_action, action.templated_command)
    pass

def _global_generation_accuracy(prediction: str, ground_truth: str)-> float:
    if prediction == ground_truth:
        return 1
    else:
        return 0


def perform_complete_behavior_generation_evaluation(model: nn.Module,
                                                    datasets: List[Tuple[str, EvalAlfredHLActionDataset]],
                                                    directory: Path) -> pd.DataFrame:
    log_directory: Path = directory / 'behavior_generation_evaluation'
    os.makedirs(log_directory, exist_ok=True)
    print("1/3 Performing forced accuracy and mrr for object selection in behavior generation...")
    with open(log_directory / 'object_forced_metrics.log') as f:
        object_forced_metrics: pd.DataFrame = apply_scoring_function_to_datasets(
            scoring_function=evaluate_object_forced_metrics_in_behavior_generation,
            datasets=datasets,
            clasp_model=model,
            log_file=f
        )
    # print("2/3 Performing generative accuracy for object selection in behavior generation...")
    # object_generative_accuracy: pd.DataFrame = apply_scoring_function_to_datasets(
    #     scoring_function=evaluate_object_generative_accuracy_in_behavior_generation,
    #     datasets=datasets,
    #     clasp_mode=model
    # )
    # print("3/3 Performing global behavior generation evaluation (quantitative and qualitative)...")
    # global_generation_accuracy: pd.DataFrame = apply_scoring_function_to_datasets(
    #     scoring_function=evaluate_global_accuracy_and_qualitative_in_behavior_generation,
    #     datasets=datasets,
    #     clasp_model=model
    # )
    all_results = pd.concat([
        object_forced_metrics,
        # object_generative_accuracy,
        # global_generation_accuracy
    ], axis=0)
    return all_results


if __name__ == '__main__':
    description = "CLASP Behavior Generation evaluation"
    args: Namespace = parse_eval_args(description)
    announce_start_evaluation(args)

    print("Loading data...")
    datasets: List[Tuple[str, EvalAlfredHLActionDataset]] = load_eval_data(args)

    checkpoint_path = get_checkpoint_path(args)
    print(f"Loading model from {checkpoint_path}...")
    if args.debug:
        clasp_model: CLASP = CLASP(512)
    else:
        clasp_model: CLASP = CLASP.load_from_checkpoint(checkpoint_path, map_location=DEVICE)

    all_results: pd.DataFrame = perform_complete_behavior_generation_evaluation(
        clasp_model, datasets, args.model_dir
    )
    print(all_results)
    save_file: Path = args.model_dir / f"behavior_generation_evaluation.csv"
    all_results.to_csv(save_file)
    print(f"Results saved to {save_file}")