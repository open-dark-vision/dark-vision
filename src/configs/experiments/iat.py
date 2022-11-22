from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    IATModelConfig,
    LOLDatasetConfig,
    Loss,
    LossConfig,
    Optimizer,
    OptimizerConfig,
    Scheduler,
    SchedulerConfig,
    Transform,
    TransformConfig,
)

iat_config = ExperimentConfig(
    name="iat-lol",
    dataset=LOLDatasetConfig(
        num_workers=12,
        pin_memory=True,
        transform=TransformConfig(
            name=Transform.FLIP, image_size=256, pair_transform=True
        ),
        batch_size=8,
    ),
    model=IATModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAMW,
        lr=2e-4,
        weight_decay=4e-4,
        scheduler=SchedulerConfig(name=Scheduler.COSINE, frequency=1),
    ),
    loss=LossConfig(name=Loss.L1, reduction="mean"),
    device="cuda",
    epochs=200,
)


iat_finetune_config = ExperimentConfig(
    name="iat-lol-finetune",
    dataset=LOLDatasetConfig(
        num_workers=12,
        pin_memory=True,
        transform=TransformConfig(name=Transform.FLIP_NO_RESIZE, pair_transform=True),
        batch_size=8,
    ),
    model=IATModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAMW,
        lr=2e-4,
        weight_decay=4e-4,
        scheduler=SchedulerConfig(name=Scheduler.COSINE, frequency=1),
    ),
    loss=LossConfig(name=Loss.L1, reduction="mean"),
    device="cuda",
    epochs=100,
)
