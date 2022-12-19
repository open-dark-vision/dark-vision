from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    LLFlowLossConfig,
    LLFlowModelConfig,
    LOLDatasetConfig,
    Optimizer,
    OptimizerConfig,
    Transform,
    TransformConfig,
)

llflow_config = ExperimentConfig(
    name="llflow-lol",
    dataset=LOLDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(name=Transform.LLFLOW, image_size=160),
        batch_size=2,
    ),
    model=LLFlowModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAM, lr=5e-4, weight_decay=0, betas=(0.9, 0.99)
    ),
    loss=LLFlowLossConfig(),
    device="cpu",
    epochs=200,
)
