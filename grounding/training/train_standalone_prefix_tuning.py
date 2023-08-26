import argparse
import os
from argparse import Namespace

import torch.multiprocessing as mp
from lightning import Trainer

from config import DEVICE
from grounding.data_processing.datasets_train import get_train_and_val_dataloaders
from grounding.models.standalone_prefix_gpt2 import StandalonePrefixTuningGPT2

parser = argparse.ArgumentParser(description='Training of a CLASP-inspired model.')

parser.add_argument('--name', type=str, help='Name of experiment')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to run')
parser.add_argument('--eval_every', type=int, default=500, help='Nb of update steps between evaluations')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers used in the data loaders.')
parser.add_argument('--gradient_clipping', type=float, default=1, help='Value of the gradient clipping')

parser.add_argument('--debug', action='store_true', help='Use very little data to debug.')
parser.add_argument('--profiler', action='store_true', help='Use a profiler to find bottlenecks in the code.')

parser.add_argument('--prefix_length', type=int, default=10, help='Number of tokens of the prefix that is tuned.')


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def launch_training(args: Namespace):
    if DEVICE == "cuda":
        mp.set_start_method("spawn")

    model = StandalonePrefixTuningGPT2(args.prefix_length, args.lr, args.weightdecay)

    train_dataloader, val_dataloader = get_train_and_val_dataloaders(
        batch_size=args.batch_size,
        clasp_mode=False,
        num_workers=args.num_workers,
        train_fraction=0.01 if args.debug else 1.
    )
    trainer: Trainer = Trainer(
        limit_train_batches=3 if args.debug else None,
        val_check_interval=args.eval_every,
        fast_dev_run=True if args.debug else False,
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clipping,     # Todo: Gradient clipping?
        detect_anomaly=True,
        profiler="simple" if args.profiler else None
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == '__main__':
    """Specify carefully z_size and prefix arguments"""
    args: Namespace = parser.parse_args()
    launch_training(args)
