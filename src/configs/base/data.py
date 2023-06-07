from dataclasses import dataclass

from omegaconf import MISSING
from typing import Optional, Tuple


from .types import PairSelectionMethod, SupplementaryDataset, Transform


@dataclass
class TransformConfig:
    name: Transform = Transform.DEVELOPMENT
    image_size: int = 256
    pair_transform: bool = False

@dataclass
class DatasetConfig:
    name: str = MISSING
    path: str = MISSING
    batch_size: int = 16
    val_size: float = 0.1
    pin_memory: bool = True
    num_workers: int = 4
    transform: TransformConfig = TransformConfig()
    predict_on_train: bool = False
    predict_on_val: bool = False


@dataclass
class ExDarkDatasetConfig(DatasetConfig):
    name: str = "ExDark"
    path: str = "data/ExDark"


@dataclass
class LOLDatasetConfig(DatasetConfig):
    name: str = "LOL"
    path: str = "data/LOL"
    path_decom: Optional[str] = "data/LOL_decom"
    preload: bool = False
    load_paths: bool = False


@dataclass
class SICEDatasetConfig(DatasetConfig):
    name: str = "SICE"
    path: str = "data/SICE"
    max_exposure_ratio: float = 1.0
    train_pair_selection_method: PairSelectionMethod = PairSelectionMethod.RANDOM_TARGET
    test_pair_selection_method: PairSelectionMethod = PairSelectionMethod.HALFEXP_TARGET


@dataclass
class SupplementaryDatasetConfig(DatasetConfig):
    name: SupplementaryDataset = MISSING
    path: str = "data/Supplementary"
