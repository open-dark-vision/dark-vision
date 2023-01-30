
from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    KinDLossConfig,
    KinDModelConfig,
    LOLDatasetConfig,
    Optimizer,
    OptimizerConfig,
    Transform,
    TransformConfig,
)

kind_config = ExperimentConfig(
    name="kind-lol",
    dataset=LOLDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(name=Transform.LLFLOW, image_size=160),
        batch_size=2,
    ),
    model=KinDModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAM, lr=5e-4, weight_decay=0, betas=(0.9, 0.99)
    ),
    loss=KinDLossConfig(),
    device="cpu",
    epochs=200,

)