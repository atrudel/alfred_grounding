import pickle
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset, DataLoader

from grounding.data_processing.action import Action
from grounding.data_processing.preprocess_data import high_level_action_dump_filename


class AlfredHLActionDataset(Dataset):
    """Dataset of high level actions used to train models in teacher forcing."""
    def __init__(self, root_dir: str, fraction: float = 1):
        self.root_dir: Path = Path(root_dir)
        with open(self.root_dir / high_level_action_dump_filename, 'rb') as f:
            full_data: List[Action] = pickle.load(f)
        self.actions: List[Action] = full_data[:int(len(full_data) * fraction)]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int):
        action: Action = self.actions[index]
        input_text = action.instruction
        input_image_feats = action.image_features
        output_text = action.templated_string
        return input_text, input_image_feats, output_text



class EvalAlfredHLActionDataset(AlfredHLActionDataset):
    pass


def get_train_and_val_dataloaders(batch_size: int, num_workers: int = 1,
                                  train_fraction: float = 1.) -> Tuple[DataLoader, DataLoader]:
    train_dataset = AlfredHLActionDataset('alfred/data/json_feat_2.1.0/train', fraction=train_fraction)
    val_seen_dataset = AlfredHLActionDataset('alfred/data/json_feat_2.1.0/valid_seen')
    print(f"Split sizes: train={len(train_dataset)}, val_seen={len(val_seen_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
    val_seen_dataloader = DataLoader(val_seen_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    return train_dataloader, val_seen_dataloader