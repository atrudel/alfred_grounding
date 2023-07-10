import argparse
import os
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.cuda
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

from grounding.data_processing.action import Action
from grounding.data_processing.datasets import EvalAlfredHLActionDataset
from grounding.data_processing.object import Object
from grounding.evaluation.scoring.forced_scoring import compute_forced_metrics_for_ambiguous_situation
from grounding.evaluation.utils import parse_eval_args, mean, object_counts
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"


class GroundingTest:
    def __init__(self,
                 object_1: str,
                 object_2: str,
                 generic_name: str,
                 object_unrelated: str,
                 instruction_template: str,
                 actions_1: List[Action],
                 actions_2: List[Action],
                 actions_both: List[Action],
                 actions_unrelated: List[Action]):
        self.object_1: Object = Object(object_1)
        self.object_2: Object = Object(object_2)
        self.generic_name: str = generic_name
        self.object_unrelated: Object = Object(object_unrelated)
        assert "{object}" in instruction_template, "Instruction template must contain the placeholder '{object}'"
        self.instruction_template: str = instruction_template
        self.situations_1: List[Action] = [action.load_image() for action in actions_1]
        self.situations_2: List[Action] = [action.load_image() for action in actions_2]
        self.situations_both: List[Action] = [action.load_image() for action in actions_both]
        self.situations_unrelated: List[Action] = [action.load_image() for action in actions_unrelated]

    def launch(self, model: ImageConditionedLLMOnDecoder) -> pd.DataFrame:
        output_objects: List[Object] = [self.object_1, self.object_2, self.object_unrelated]
        mrr_column_names: List[str] = [f"MRR_{object.name}" for object in output_objects]
        top_object_column_names: List[str] = ["Top_object"]
        denominations: List[str] = [self.object_1.name, self.object_2.name, self.generic_name]
        image_contents: List[str] = [self.object_1.name, self.object_2.name, 'both', self.object_unrelated.name]

        # Create result dataframe
        index: pd.MultiIndex = pd.MultiIndex.from_product([denominations, image_contents],
                                                          names=['Instruction', 'Image'])
        results: pd.DataFrame = pd.DataFrame(index=index, columns=mrr_column_names + top_object_column_names)

        for instruct_object_name in tqdm(denominations):
            instruction = self.instruction_template.format(object=instruct_object_name)
            for situation_set, image_content in zip(
                    [self.situations_1, self.situations_2, self.situations_both, self.situations_unrelated],
                    image_contents):
                rrs_obj1 = []
                rrs_obj2 = []
                rrs_unrelated = []
                top_objects = []
                for situation in situation_set:
                    situation.instruction = instruction
                    (rr_obj1, rr_obj2, rr_unrelated), top_object = \
                        compute_forced_metrics_for_ambiguous_situation(situation, model, output_objects)
                    rrs_obj1.append(rr_obj1)
                    rrs_obj2.append(rr_obj2)
                    rrs_unrelated.append(rr_unrelated)
                    top_objects.append(top_object)
                results.loc[(instruct_object_name, image_content), mrr_column_names[0]] = mean(rrs_obj1)
                results.loc[(instruct_object_name, image_content), mrr_column_names[1]] = mean(rrs_obj2)
                results.loc[(instruct_object_name, image_content), mrr_column_names[2]] = mean(rrs_unrelated)
                results.loc[(instruct_object_name, image_content), top_object_column_names[0]] = object_counts(top_objects)

        return results


    def show(self):
        n_cols = 2
        imgsize = 30

        def plot_images(actions: List[Action], title: str):
            n_rows = len(actions) // n_cols + 1
            fig = plt.figure(figsize=(n_cols * imgsize, n_rows * imgsize))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                             axes_pad=0.1,  # pad between axes in inch.
                             )
            for ax, action in zip(grid, actions):
                img = action.get_image()
                ax.imshow(img)
            plt.suptitle(title, fontsize=90)
            plt.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
            plt.show()

        action_sets = [self.situations_1, self.situations_2, self.situations_both, self.situations_unrelated]
        titles = [
            f"Examples with {self.object_1.name}",
            f"Examples with {self.object_2.name}",
            f"Examples with {self.object_1.name} and {self.object_2.name}",
            f"Examples with {self.object_unrelated.name}"
        ]
        for actions, title in zip(action_sets, titles):
            plot_images(actions, title)

    def save(self):
        save_dir = Path('eval_datasets')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"grounding-test_{self.object_1.name}-{self.object_2.name}.pickle"
        filepath = save_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Grounding test saved to {filepath}")

    def __str__(self) -> str:
        return f"grounding_test({self.object_1.name},{self.object_2.name}||{self.object_unrelated.name})"


if __name__ == '__main__':
    args: argparse.Namespace = parse_eval_args()

    print("Loading data...")
    data = EvalAlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_unseen')
    grounding_test = GroundingTest(
        object_1='tomato',
        object_2='lettuce',
        generic_name='vegetable',
        object_unrelated='vase',
        instruction_template="Grab the {object} in front of you.",
        actions_1=data.get_actions_by_indices([112, 1092, 3770, 4474, 4928]),
        actions_2=data.get_actions_by_indices([310, 621, 2001, 2968]),
        actions_both=data.get_actions_by_indices([876, 882, 2516, 459, 3311, 4510]),
        actions_unrelated=data.get_actions_by_indices([4627, 1230, 1566, 2986]),
    )
    print("Loading model...")
    model = ImageConditionedLLMOnDecoder.load(args.model_path).to(device)
    print("Launching test")
    results: pd.DataFrame = grounding_test.launch(model)
    save_path: Path = args.model_dir / f"{str(grounding_test)}.csv"
    results.to_csv(save_path)
    print(f"Results saved to {save_path}")

