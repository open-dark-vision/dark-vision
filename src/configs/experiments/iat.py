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
    name="iat",
    dataset=LOLDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(
            name=Transform.IAT, image_size=256, pair_transform=True
        ),
        batch_size=4,
    ),
    model=IATModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAMW,
        lr=1e-3,
        weight_decay=1e-4,
        scheduler=SchedulerConfig(
            name=Scheduler.ONE_CYCLE, interval="step", frequency=1
        ),
    ),
    loss=LossConfig(name=Loss.L1, reduction="mean"),
    device="cpu",
    epochs=3,
)
