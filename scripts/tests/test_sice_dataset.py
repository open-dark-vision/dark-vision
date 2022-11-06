"""
Run it to make sure that your SICE dataset configs are ok.
"""
import argparse
from timeit import default_timer as timer

from omegaconf import OmegaConf

from src.datamodules import SICEDataModule  # noqa: I900

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/tests/sice_dataset_test.yaml"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print("Configs:\n", OmegaConf.to_yaml(conf), "", sep="*" * 50 + "\n")

    setup_timer_start = timer()
    sice_dm = SICEDataModule(conf["dataset"])
    sice_dm.setup()
    setup_timer_stop = timer()

    data_loader = iter(sice_dm.train_dataloader())

    get_timer_start = timer()
    batch = next(data_loader)
    get_timer_stop = timer()

    print(f"Setup datamodule time: {setup_timer_stop - setup_timer_start:.4f} sec")
    print(f"Get single batch time: {get_timer_stop - get_timer_start:.4f} sec")
    print(f"Image batch size:  {batch['image'].size()}")
    print(f"Target batch size: {batch['target'].size()}")
