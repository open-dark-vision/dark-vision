from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ModelConfig:
    name: str = MISSING
