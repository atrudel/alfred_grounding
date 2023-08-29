import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

from config import DEVICE
from grounding.evaluation.grounding_capability.grounding_test import GroundingTest, build_grounding_tests
from grounding.evaluation.utils import parse_eval_args
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder


if __name__ == '__main__':
    args: argparse.Namespace = parse_eval_args()
    save_dir: Path = args.model_dir / 'grounding_evaluation'
    os.makedirs(save_dir, exist_ok=True)

    print("Loading grounding tests...")
    grounding_tests: List[GroundingTest] = build_grounding_tests()

    print("Loading model...")
    model = ImageConditionedLLMOnDecoder.load(args.model_path).to(DEVICE)
    model.eval()

    for grounding_test in grounding_tests:
        print(f"Launching grounding test: {grounding_test.title}")
        results: pd.DataFrame = grounding_test.launch(model)
        save_path: Path = save_dir / f"{str(grounding_test)}.csv"
        results.to_csv(save_path)
        print(f"Results saved to {save_path}")

