from src.configs.base import (  # noqa: I900
    ExDarkDatasetConfig,
    ExperimentConfig,
    LOLDatasetConfig,
    SICEDatasetConfig,
    SupplementaryDatasetConfig,
    TransformConfig,
)

exdark_dataset_test_config = ExperimentConfig(
    name="ExDark Dataset Test",
    dataset=ExDarkDatasetConfig(
        num_workers=0,
        pin_memory=False,
    ),
)


lol_dataset_test_config = ExperimentConfig(
    name="LOL Dataset Test",
    dataset=LOLDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(pair_transform=True),
    ),
)


sice_dataset_test_config = ExperimentConfig(
    name="SICE Dataset Test",
    dataset=SICEDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(pair_transform=True),
    ),
)


supplementary_dataset_test_config = ExperimentConfig(
    name="Supplementary Dataset Test",
    dataset=SupplementaryDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(
            image_size=128,
        ),
    ),
)
