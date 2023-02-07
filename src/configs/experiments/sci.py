from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    Loss,
    LossConfig,
    Optimizer,
    OptimizerConfig,
    PairSelectionMethod,
    Scheduler,
    SchedulerConfig,
    SCIModelConfig,
    SICEDatasetConfig,
    Transform,
    TransformConfig,
)

sci_config = ExperimentConfig(
    name="sci-sice",
    finetune=False,
    dataset=SICEDatasetConfig(
        path="data/MiniSICE",
        num_workers=12,
        pin_memory=True,
        transform=TransformConfig(
            name=Transform.FLIP_CENTER_CROP, image_size=256, pair_transform=True
        ),
        batch_size=16,
        train_pair_selection_method=PairSelectionMethod.RANDOM_HALFEXP,
        test_pair_selection_method=PairSelectionMethod.DARKEST_HALFEXP,
        max_exposure_ratio=0.75,
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
    epochs=500,
)

sci_finetune_config = ExperimentConfig(
    name="sci-sice-finetune",
    finetune=True,
    dataset=SICEDatasetConfig(
        path="data/MiniSICE",
        num_workers=12,
        pin_memory=True,
        transform=TransformConfig(
            name=Transform.FLIP_CENTER_CROP, image_size=512, pair_transform=True
        ),
        batch_size=16,
        train_pair_selection_method=PairSelectionMethod.RANDOM_HALFEXP,
        test_pair_selection_method=PairSelectionMethod.DARKEST_HALFEXP,
    ),
    model=SCIModelConfig(
        supervised_metrics=True,
    ),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAMW,
        lr=3e-4,
        weight_decay=1e-2,
        scheduler=SchedulerConfig(name=Scheduler.COSINE, frequency=1),
    ),
    loss=LossConfig(
        name=Loss.SCI,
    ),
    device="cuda",
    epochs=100,
)
