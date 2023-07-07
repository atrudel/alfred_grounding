import os
import pickle
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from grounding.data_processing.action import Action


class GroundingTest:
    def __init__(self,
                 object_1: str,
                 object_2: str,
                 generic: str,
                 actions_1: List[Action],
                 actions_2: List[Action],
                 actions_both: List[Action],
                 actions_unrelated: List[Action]):
        self.object_1: str = object_1
        self.object_2: str = object_2
        self.actions_1: List[Action] = [action.load_image() for action in actions_1]
        self.actions_2: List[Action] = [action.load_image() for action in actions_2]
        self.actions_both: List[Action] = [action.load_image() for action in actions_both]
        self.actions_unrelated: List[Action] = [action.load_image() for action in actions_unrelated]
        self.instruction_1: str = self.actions_1[0].instruction
        self.instruction_2: str = self.actions_2[0].instruction

    def show(self):
        n_cols = 4

        def plot_images(actions: List[Action], title: str):
            n_rows = len(actions) // n_cols + 1
            fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                             nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                             axes_pad=0.1,  # pad between axes in inch.
                             )
            for ax, action in zip(grid, actions):
                img = action.get_image()
                ax.imshow(img)
            plt.suptitle(title, fontsize='xx-large')
            plt.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
            plt.show()

        action_sets = [self.actions_1, self.actions_2, self.actions_both, self.actions_unrelated]
        titles = [
            f"Examples with {self.object_1}",
            f"Examples with {self.object_2}",
            f"Examples with {self.object_1} and {self.object_2}",
            f"Examples with an unrelated object"
        ]
        for actions, title in zip(action_sets, titles):
            plot_images(actions, title)

    def save(self):
        save_dir = Path('eval_datasets')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"grounding-test_{self.object_1}-{self.object_2}.pickle"
        filepath = save_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Grounding test saved to {filepath}")


