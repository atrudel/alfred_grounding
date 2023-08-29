import optuna
from lightning import Trainer
from optuna import Trial
from optuna.integration import PyTorchLightningPruningCallback

from data_processing.datasets_train import get_train_and_val_dataloaders
from models.clasp import CLASP


def objective(trial: Trial):
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
        weightdecay=weightdecay
    )
    train_dataloader, val_dataloader = get_train_and_val_dataloaders(
        batch_size=12,
        clasp_mode=True,
        num_workers=1,
        train_fraction=1
    )

    trainer = Trainer(
        max_epochs=50,
        gradient_clip_val=gradient_clipping,
    )
    trainer.fit(
        model=clasp_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    val_loss = trainer.logged_metrics["val_loss"]
    return val_loss


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)