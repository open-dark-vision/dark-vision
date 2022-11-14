from src.configs.base import ExDarkDatasetConfig, ExperimentConfig  # noqa: I900

exdark_dataset_test_config = ExperimentConfig(
    name="ExDark Dataset Test",
    dataset=ExDarkDatasetConfig(
        num_workers=0,
        pin_memory=False,
    ),
)
