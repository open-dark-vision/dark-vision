"""
Run it to make sure that your ExDark dataset configs are ok.
"""
from timeit import default_timer as timer

from omegaconf import OmegaConf

from src.configs.tests import exdark_dataset_test_config as cfg  # noqa: I900
from src.datamodules import ExDarkDataModule  # noqa: I900

if __name__ == "__main__":
    cfg = OmegaConf.structured(cfg)
    print("Configs:\n", OmegaConf.to_yaml(cfg), "", sep="*" * 50 + "\n")

    setup_timer_start = timer()
    exdark_dm = ExDarkDataModule(cfg["dataset"])
    exdark_dm.setup()
    setup_timer_stop = timer()

    data_loader = iter(exdark_dm.train_dataloader())

    get_timer_start = timer()
    batch = next(data_loader)
    get_timer_stop = timer()

    print(f"Setup datamodule time: {setup_timer_stop - setup_timer_start:.4f} sec")
    print(f"Get single batch time: {get_timer_stop - get_timer_start:.4f} sec")
    print(f"Image batch size:  {batch['image'].size()}")
    print(f"Labels batch size: {len(batch['labels'])}")
    print(f"BBoxes batch size: {len(batch['bboxes'])}")
