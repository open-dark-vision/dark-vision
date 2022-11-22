from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    LOLDatasetConfig,
    Loss,
    LossConfig,
    Optimizer,
    OptimizerConfig,
    Scheduler,
    SchedulerConfig,
    SCIModelConfig,
    Transform,
    TransformConfig,
)

sci_config = ExperimentConfig(
    name="sci-lol",
    finetune=False,
    dataset=LOLDatasetConfig(
        num_workers=12,
        pin_memory=True,
        transform=TransformConfig(
            name=Transform.FLIP, image_size=256, pair_transform=True
        ),
        batch_size=8,
    ),
    model=SCIModelConfig(
        supervised_metrics=True,
    ),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAMW,
        lr=3e-4,
        weight_decay=1e-2,
        scheduler=SchedulerConfig(name=Scheduler.CONSTANT, frequency=1),
    ),
    loss=LossConfig(
        name=Loss.SCI,
    ),
    device="cuda",
    epochs=30,
)

sci_finetune_config = ExperimentConfig(
    name="sci-lol-finetune",
    finetune=True,
    dataset=LOLDatasetConfig(
        num_workers=12,
        pin_memory=True,
        transform=TransformConfig(name=Transform.FLIP_NO_RESIZE, pair_transform=True),
        batch_size=8,
    ),
    model=SCIModelConfig(
        supervised_metrics=True,
    ),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAMW,
        lr=5e-4,
        weight_decay=1e-2,
        scheduler=SchedulerConfig(name=Scheduler.CONSTANT, frequency=1),
    ),
    loss=LossConfig(
        name=Loss.SCI,
    ),
    device="cuda",
    epochs=30,
)
