"""
Run it to make sure that your Supplemntary dataset configs are ok.
"""
import argparse
from timeit import default_timer as timer

from omegaconf import OmegaConf

from src.datamodules import SupplementaryDataModule  # noqa: I900
from src.datasets.supplementary import SupplementaryNames  # noqa: I900

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/tests/supplementary_dataset_test.yaml"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print("Configs:\n", OmegaConf.to_yaml(conf), "", sep="*" * 50 + "\n")

    for dataset in SupplementaryNames:
        print(f"Testing {dataset.name} dataset")
        conf["dataset"]["name"] = dataset.value

        setup_timer_start = timer()
        supplementary_dm = SupplementaryDataModule(conf["dataset"])
        supplementary_dm.setup()
        setup_timer_stop = timer()

        data_loader = iter(supplementary_dm.test_dataloader())

        get_timer_start = timer()
        batch = next(data_loader)
        get_timer_stop = timer()

        print(f"Setup datamodule time: {setup_timer_stop - setup_timer_start:.4f} sec")
        print(f"Get single batch time: {get_timer_stop - get_timer_start:.4f} sec")
        print(f"Image batch size: {batch['image'].size()} \n")
