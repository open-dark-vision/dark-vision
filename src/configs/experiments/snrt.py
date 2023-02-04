from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    LOLDatasetConfig,
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
        transform=TransformConfig(name=Transform.FLIP_NO_RESIZE, pair_transform=True),
        batch_size=8,
    ),
    model=SNRTModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAMW,
        lr=2e-4,
        weight_decay=4e-4,
        scheduler=SchedulerConfig(name=Scheduler.COSINE, frequency=1),
    ),
    device="mps",
    epochs=1,
)
