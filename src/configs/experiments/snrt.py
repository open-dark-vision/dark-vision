from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    LOLDatasetConfig,
    Loss,
    LossConfig,
    Optimizer,
    OptimizerConfig,
    Scheduler,
    SchedulerConfig,
    SNRTModelConfig,
    Transform,
    TransformConfig,
)

snrt_config = ExperimentConfig(
    name="snrt-lol",
    dataset=LOLDatasetConfig(
        num_workers=12,
        pin_memory=True,
        transform=TransformConfig(
            name=Transform.FLIP_NO_SCALE, pair_transform=True, image_size=128
        ),
        batch_size=32,
    ),
    model=SNRTModelConfig(
        lambd=0.1,
        # lambd=0.02,
    ),
    loss=LossConfig(name=Loss.CHAR_VGG),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAMW,
        lr=4e-4,
        weight_decay=4e-4,
        scheduler=SchedulerConfig(name=Scheduler.COSINE, frequency=1),
    ),
    device="cuda",
    epochs=100,
)
