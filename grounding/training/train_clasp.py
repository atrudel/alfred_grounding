import argparse
from argparse import Namespace

from lightning import Trainer

from grounding.data_processing.datasets import get_train_and_val_dataloaders
from grounding.models.clasp.clasp import CLASP

parser = argparse.ArgumentParser(description='Training of a CLASP-inspired model.')

parser.add_argument('--name', type=str, help='Name of experiment')
parser.add_argument('--no_image', action='store_true')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to run')
parser.add_argument('--eval_every', type=int, default=200, help='Nb of update steps between evaluations')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers used in the data loaders.')

parser.add_argument('--z_size', type=int, default=512, help='Size of the z embedding.')
parser.add_argument('--prefix', action='store_true', help='Use prefix tuning in decoding z to text.')
parser.add_argument('--beta_caption', type=float, default=1., help='Coefficient for the captioning loss component.')
parser.add_argument('--beta_behav_gen', type=float, default=1., help='Coefficient for the behavior generation loss component.')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature use in the contrastive loss.')


def launch_training(args: Namespace):
    clasp_model = CLASP(z_size=args.z_size,
                        prefix_tuning=args.prefix,
                        beta_align=1,
                        beta_caption=args.beta_caption,
                        beta_behavior_gen=args.beta_behav_gen,
                        temperature=args.temperature,
                        learning_rate=args.lr,
                        weightdecay=args.weightdecay
                        )
    train_dataloader, val_dataloader = get_train_and_val_dataloaders(
        batch_size=args.batch_size,
        use_raw_images=True,
        num_workers=2,
        train_fraction=0.01 if args.debug else 1.
    )
    trainer: Trainer = Trainer(
        limit_train_batches=2 if args.debug else None,
        val_check_interval=args.eval_every,
        fast_dev_run=True if args.debug else False,
        max_epochs=args.epochs,
        # gradient_clip_val=1.,     #Todo: Gradient clipping?
        detect_anomaly=True
    )
    trainer.fit(
        model=clasp_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == '__main__':
    """Specify carefully z_size and prefix arguments"""
    args: Namespace = parser.parse_args()
    launch_training(args)