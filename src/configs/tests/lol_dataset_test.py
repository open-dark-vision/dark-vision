from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    LOLDatasetConfig,
    TransformConfig,
)

lol_dataset_test_config = ExperimentConfig(
    name="LOL Dataset Test",
    dataset=LOLDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(pair_transform=True),
    ),
)
