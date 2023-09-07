import argparse
from argparse import Namespace

import torch.multiprocessing as mp
from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary

from config import DEVICE, REPO_ROOT
from grounding.data_processing.datasets_train import get_train_and_val_dataloaders
from grounding.models.clasp import CLASP

parser = argparse.ArgumentParser(description='Training of a CLASP-inspired model.')

parser.add_argument('--name', type=str, help='Name of experiment')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to run')
parser.add_argument('--eval_every', type=float, default=0.25, help='Interval between validations as a fraction of the epoch.')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in the data loaders.')
parser.add_argument('--gradient_clipping', type=float, default=1., help='Value of the gradient clipping')

parser.add_argument('--debug', action='store_true', help='Use very little data to debug.')
parser.add_argument('--overfit', action='store_true', help='Overfit a small portion of the training data to debug.')
parser.add_argument('--profiler', action='store_true', help='Use a profiler to find bottlenecks in the code.')

parser.add_argument('--z_size', type=int, default=512, help='Size of the z embedding.')
parser.add_argument('--beta_caption', type=float, default=1., help='Coefficient for the captioning loss component.')
parser.add_argument('--beta_behav_gen', type=float, default=1., help='Coefficient for the behavior generation loss component.')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature use in the contrastive loss.')


def launch_training(args: Namespace):
    if DEVICE == "cuda":
        mp.set_start_method("spawn")

    clasp_model = CLASP(z_size=args.z_size, beta_align=1, beta_caption=args.beta_caption,
                        beta_behavior_gen=args.beta_behav_gen, temperature=args.temperature, learning_rate=args.lr,
                        weightdecay=args.weightdecay)
    train_dataloader, val_dataloader = get_train_and_val_dataloaders(
        batch_size=args.batch_size,
        clasp_mode=True,
        num_workers=args.num_workers,
        train_fraction=1
    )
    trainer: Trainer = Trainer(
        limit_train_batches=3 if args.debug else None,
        val_check_interval=1 if args.overfit else args.eval_every,
        fast_dev_run=True if args.debug else False,
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clipping,
        detect_anomaly=True,
        profiler="advanced" if args.profiler else None,
        overfit_batches=1 if args.overfit else 0.,
        log_every_n_steps=1 if args.overfit else 50,
        num_sanity_val_steps=0 if (args.debug or args.overfit) else 2,
        callbacks=[ModelSummary(max_depth=3)],
        default_root_dir=REPO_ROOT / 'debug' if args.debug else None
    )
    trainer.fit(
        model=clasp_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == '__main__':
    """Specify carefully z_size and prefix arguments"""
    args: Namespace = parser.parse_args()
    launch_training(args)
