from dataclasses import dataclass

from omegaconf import MISSING

from .data import DatasetConfig
from .model import ModelConfig
from .training import LossConfig, OptimizerConfig, SchedulerConfig


@dataclass
class ExperimentConfig:
    name: str = MISSING

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    scheduler: SchedulerConfig = MISSING
    loss: LossConfig = MISSING

    seed: int = 42
    device: str = "cuda"
    log_dir: str = "wandb"
    checkpoint_dir: str = "checkpoints"
