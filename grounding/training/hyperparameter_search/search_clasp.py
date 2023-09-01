import argparse
from argparse import Namespace
from datetime import datetime
from typing import List, Dict

from torch import Tensor

from config import REPO_ROOT
import optuna
import torch.multiprocessing as mp
from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary
from optuna import Trial

from config import DEVICE
from grounding.data_processing.datasets_train import get_train_and_val_dataloaders
from grounding.models.clasp import CLASP

parser = argparse.ArgumentParser(description='Training of a CLASP-inspired model.')
parser.add_argument('--debug', action='store_true', help='Use very little data to debug.')


def objective(trial: Trial) -> float:
    z_size = trial.suggest_int("z_size", 20, 512)
    temperature = trial.suggest_float("temperature", 0.01, 5)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
    weightdecay = trial.suggest_float("weightdecay", 1e-5, 1e-3, log=True)
    gradient_clipping = trial.suggest_float("gradient_clipping", 1., 10.)

    clasp_model = CLASP(
        z_size=z_size,
        beta_align=1,
        beta_caption=1,
        beta_behavior_gen=1,
        temperature=temperature,
        learning_rate=learning_rate,
        weightdecay=weightdecay,
    )
    train_dataloader, val_dataloader = get_train_and_val_dataloaders(
        batch_size=12,
        clasp_mode=True,
        num_workers=4,
        train_fraction=1
    )

    trainer: Trainer = Trainer(
        enable_checkpointing=False,
        enable_model_summary=False,
        max_epochs=1 if args.debug else 20,
        val_check_interval=0.4,
        accelerator="auto",
        devices=1,
        gradient_clip_val=gradient_clipping,
        limit_train_batches=0.01 if args.debug else 1.0,
        limit_val_batches=0.5 if args.debug else 1.0,
        default_root_dir=REPO_ROOT / 'hparam_search',
        num_sanity_val_steps=0 if args.debug else 2
    )
    hyperparameters = dict(
        z_size=z_size,
        temperature=temperature,
        learning_rate=learning_rate,
        weightdecay=weightdecay,
        gradient_clipping=gradient_clipping
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(
        model=clasp_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    val_loss: List[Dict[str, float]] = trainer.validate(clasp_model, dataloaders=val_dataloader)
    return val_loss[0]['val_loss']


if __name__ == '__main__':
    args: Namespace = parser.parse_args()

    if DEVICE == "cuda":
        mp.set_start_method("spawn")
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage="sqlite:///db.sqlite3",
        study_name=f"hparam_search_{datetime.now()}"
    )
    study.optimize(objective, n_trials=50)
    print(f"Best value: {study.best_value} (params: {study.best_params})")