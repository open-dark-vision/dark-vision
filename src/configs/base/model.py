from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = MISSING


@dataclass
class IATModelConfig(ModelConfig):
    name: str = "IAT"
    in_dim: int = 3
    task_type: str = "lol"
    layers_type: str = "cct"
