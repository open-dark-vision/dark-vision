from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf import MISSING

from .types import Loss, Optimizer, Scheduler


@dataclass
class SchedulerConfig:
    name: Scheduler = MISSING
    frequency: int = MISSING


@dataclass
class OptimizerConfig:
    name: Optimizer = MISSING
    lr: float = 1e-3
    weight_decay: float = 1e-2
    scheduler: Optional[SchedulerConfig] = None
    betas: Optional[Tuple] = None


@dataclass
class LossConfig:
    name: Loss = MISSING
    reduction: str = "mean"


@dataclass
class LLFlowLossConfig(LossConfig):
    name: Loss = Loss.NLL
    p: float = 0.2

@dataclass
class KinDLossConfig(LossConfig):
    name: Loss = Loss.KIND
    p: float = 0.2




