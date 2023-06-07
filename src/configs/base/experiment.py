from dataclasses import dataclass

from omegaconf import MISSING

from .data import DatasetConfig
from .model import ModelConfig
from .training import LossConfig, OptimizerConfig
from typing import Optional

@dataclass
class ExperimentConfig:
    name: str = MISSING
    model_name: Optional[str] = None
    save_predictions: Optional[str] = False
    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    optimizer: Optional[OptimizerConfig] = MISSING
    optimizer_restoration: Optional[OptimizerConfig] = MISSING
    checkpoint_path: Optional[str] = None
    loss: LossConfig = MISSING

    finetune: bool = False  # use when you want to have 2 stages of training
    seed: int = 42
    epochs: int = 10
    device: str = "cuda"
