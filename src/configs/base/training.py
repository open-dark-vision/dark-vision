from dataclasses import dataclass

from omegaconf import MISSING

from .types import Optimizer, Scheduler


@dataclass
class OptimizerConfig:
    name: Optimizer = MISSING
    lr: float = 1e-3
    weight_decay: float = 1e-2


@dataclass
class SchedulerConfig:
    name: Scheduler = MISSING
    interval: str = MISSING
    frequency: int = MISSING


@dataclass
class LossConfig:
    name: str = MISSING  # TODO: what losses do we have?
    reduction: str = "mean"
