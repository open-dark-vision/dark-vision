"""
Run it to make sure that your LOL dataset configs are ok.
"""
from timeit import default_timer as timer

from omegaconf import OmegaConf

from src.configs.tests import lol_dataset_test_config as cfg  # noqa: I900
from src.datamodules import LOLDataModule  # noqa: I900

if __name__ == "__main__":
    cfg = OmegaConf.structured(cfg)
    print("Configs:\n", OmegaConf.to_yaml(cfg), "", sep="*" * 50 + "\n")

    setup_timer_start = timer()
    lol_dm = LOLDataModule(cfg["dataset"])
    lol_dm.setup()
    setup_timer_stop = timer()

    data_loader = iter(lol_dm.train_dataloader())

    get_timer_start = timer()
    batch = next(data_loader)
    get_timer_stop = timer()

    print(f"Setup datamodule time: {setup_timer_stop - setup_timer_start:.4f} sec")
    print(f"Get single batch time: {get_timer_stop - get_timer_start:.4f} sec")
    print(f"Image batch size:  {batch['image'].size()}, type {batch['image'].dtype}")
    print(f"Target batch size: {batch['target'].size()}, type {batch['target'].dtype}")
