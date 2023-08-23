import os
import pickle
from pathlib import Path
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from config import REPO_ROOT
from data_processing.preprocess_data import PREPROCESSED_ACTIONS_DIR_NAME
from grounding.data_processing.action import Action


class AlfredHLActionDataset(Dataset):
    """Dataset of high level actions used to train models in teacher forcing."""
    def __init__(self, root_dir: str, clasp_mode: bool = False, fraction: float = 1, debug: bool = False):
        super().__init__()
        self.root_dir: Path = Path(root_dir) / PREPROCESSED_ACTIONS_DIR_NAME

        print(f"Loading dataset from {root_dir}...", end='')
        all_filenames = sorted(os.listdir(self.root_dir))
        self.filenames = all_filenames[:int(len(all_filenames) * fraction)]
        self.debug: bool = debug
        self.clasp_mode: bool = clasp_mode

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        action = self.get_action(index)
        if self.clasp_mode:
            return self._configure_action_info_for_clasp(action)
        else:
            return self._configure_action_info_for_baseline(action)

    def get_action(self, index) -> Action:
        filename: str = self.filenames[index]
        with open(self.root_dir / filename, 'rb') as f:
            action: Action = pickle.load(f)
        return action

    def _configure_action_info_for_baseline(self, action: Action) -> Tuple[str, Tensor, str]:
        instruction: str = action.instruction
        image_resnet_feats: Tensor = action.image_resnet_features
        command: str = action.templated_command
        return instruction, image_resnet_feats, command

    def _configure_action_info_for_clasp(self, action: Action):
        return {
            # Instruction
            "instruction": action.instruction,
            "instruction_clip_feats": action.instruction_clip_features,
            # Image
            "image_clip_feats": action.image_clip_features,
            # Command
            "command": action.templated_command,
            "command_clip_feats": action.command_clip_features
        }


def get_train_and_val_dataloaders(batch_size: int, clasp_mode: bool = False, num_workers: int = 1,
                                  train_fraction: float = 1.) -> Tuple[DataLoader, DataLoader]:
    train_dataset = AlfredHLActionDataset(REPO_ROOT / 'alfred/data/json_feat_2.1.0/train',
                                          clasp_mode=clasp_mode, fraction=train_fraction)
    val_seen_dataset = AlfredHLActionDataset(REPO_ROOT / 'alfred/data/json_feat_2.1.0/valid_seen',
                                             clasp_mode=clasp_mode)
    print(f"Split sizes: train={len(train_dataset)}, val_seen={len(val_seen_dataset)}")

    train_dataloader = DataLoader(train_dataset,
                                  # collate_fn=collate_with_ if clasp_mode else None,
                                  batch_size=batch_size, num_workers=num_workers,
                                  drop_last=True, shuffle=True)
    val_seen_dataloader = DataLoader(val_seen_dataset,
                                     # collate_fn=collate_with_ if clasp_mode else None,
                                     batch_size=batch_size, num_workers=num_workers,
                                     drop_last=True)
    return train_dataloader, val_seen_dataloader

if __name__ == '__main__':
    train_dataloader, val_seen_dataloader = get_train_and_val_dataloaders(
        batch_size=12,
        clasp_mode=False,
        num_workers=1,
        train_fraction=1
    )
    action_0 = train_dataloader.dataset[0]
    a=1