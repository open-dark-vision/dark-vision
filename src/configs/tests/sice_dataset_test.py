from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    SICEDatasetConfig,
    TransformConfig,
)

sice_dataset_test_config = ExperimentConfig(
    name="SICE Dataset Test",
    dataset=SICEDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(pair_transform=True),
    ),
)
