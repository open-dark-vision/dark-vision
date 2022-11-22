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


@dataclass
class LLFlowEncoderConfig(ModelConfig):
    name: str = "LLFlow Conditional Encoder"
    channels_in: int = 12
    channels_middle: int = 64
    channels_out: int = 3
    rrdb_number: int = 24
    rrdb_channels: int = 32


@dataclass
class LLFlowModelConfig(ModelConfig):
    name: str = "LLFlow"
    encoder: ModelConfig = LLFlowEncoderConfig
