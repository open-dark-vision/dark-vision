from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    SupplementaryDatasetConfig,
    TransformConfig,
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
