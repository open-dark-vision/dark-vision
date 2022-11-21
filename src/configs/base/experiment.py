from dataclasses import dataclass

from omegaconf import MISSING

from .data import DatasetConfig
from .model import ModelConfig
from .training import LossConfig, OptimizerConfig


@dataclass
class ExperimentConfig:
    name: str = MISSING

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    loss: LossConfig = MISSING

    seed: int = 42
    epochs: int = 10
    device: str = "cuda"
