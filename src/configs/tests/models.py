from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    LLFlowEncoderConfig,
    LOLDatasetConfig,
    Transform,
    TransformConfig,
)

llflow_encoder_test_config = ExperimentConfig(
    name="LLFlow Conditional Encoder test",
    model=LLFlowEncoderConfig(),
    dataset=LOLDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(name=Transform.LLFLOW, image_size=160),
        batch_size=8,
    ),
)
