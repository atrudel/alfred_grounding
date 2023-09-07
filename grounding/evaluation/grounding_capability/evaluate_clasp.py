import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

from config import DEVICE
from grounding.evaluation.grounding_capability.utils import get_model_path
from grounding.evaluation.grounding_capability.grounding_test import GroundingTest, build_grounding_tests
from grounding.evaluation.utils import parse_eval_args
from grounding.models.clasp import CLASP


if __name__ == '__main__':
    args: argparse.Namespace = parse_eval_args()
    save_dir: Path = args.model_dir / 'grounding_evaluation'
    os.makedirs(save_dir, exist_ok=True)

    model_path: Path = get_model_path(args)

    print("Loading grounding tests...")
    grounding_tests: List[GroundingTest] = build_grounding_tests()

    print(f"Loading model fom {model_path}...")
    model = CLASP.load_from_checkpoint(model_path, map_location=DEVICE)
    model.eval()

    for grounding_test in grounding_tests:
        print(f"Launching grounding test: {grounding_test.title}")
        results: pd.DataFrame = grounding_test.launch(model)
        save_path: Path = save_dir / f"{str(grounding_test)}.csv"
        results.to_csv(save_path)
        print(f"Results saved to {save_path}")