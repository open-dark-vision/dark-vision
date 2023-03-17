from dataclasses import dataclass
from typing import Optional

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
    channels_middle: int = 32
    channels_out: int = 3
    rrdb_number: int = 4
    rrdb_channels: int = 32


@dataclass
class LLFlowInvertibleNetworkConfig(ModelConfig):
    name: str = "LLFlow Invertible Network"
    n_levels: int = 3
    n_flow_steps: int = 4
    n_additional_steps: int = 2
    n_conditional_channels: int = 32  # has to be the same as channels_middle in encoder


@dataclass
class LLFlowModelConfig(ModelConfig):
    name: str = "LLFlow"
    encoder: ModelConfig = LLFlowEncoderConfig
    flow: ModelConfig = LLFlowInvertibleNetworkConfig


@dataclass
class SCIModelConfig(ModelConfig):
    name: str = "SCI"
    stage: int = 3
    supervised_metrics: bool = False


@dataclass
class SNRTModelConfig(ModelConfig):
    name: str = "SNRT"
    nf: int = 64
    nframes: int = 5
    front_RBs: int = 1
    back_RBs: int = 1
    center: Optional[int] = None
    predeblur: bool = True
    w_TSA: bool = True
    lambd: float = 0.1
    l_pix_w: int = 1
    blur_kernel: int = 5
