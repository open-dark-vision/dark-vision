"""
Run it to make sure that your Supplemntary dataset configs are ok.
"""
from timeit import default_timer as timer

from omegaconf import OmegaConf

from src.configs.base import SupplementaryDataset  # noqa: I900
from src.configs.tests import supplementary_dataset_test_config as cfg  # noqa: I900
from src.datamodules import SupplementaryDataModule  # noqa: I900

if __name__ == "__main__":
    cfg = OmegaConf.structured(cfg)
    print("Configs:\n", OmegaConf.to_yaml(cfg), "", sep="*" * 50 + "\n")

    for dataset in SupplementaryDataset:
        print(f"Testing {dataset.name} dataset")
        cfg["dataset"]["name"] = dataset

        setup_timer_start = timer()
        supplementary_dm = SupplementaryDataModule(cfg["dataset"])
        supplementary_dm.setup()
        setup_timer_stop = timer()

        data_loader = iter(supplementary_dm.test_dataloader())

        get_timer_start = timer()
        batch = next(data_loader)
        get_timer_stop = timer()

        print(f"Setup datamodule time: {setup_timer_stop - setup_timer_start:.4f} sec")
        print(f"Get single batch time: {get_timer_stop - get_timer_start:.4f} sec")
        print(f"Image batch size: {batch['image'].size()} \n")
