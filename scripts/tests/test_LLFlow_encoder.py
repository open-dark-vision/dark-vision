"""
Run it to make sure that your LLFlow conditional encoder configs are ok.
"""
from omegaconf import OmegaConf

from src.configs.tests import llflow_encoder_test_config as cfg  # noqa: I900
from src.datasets import LOLDataModule  # noqa: I900
from src.models import ConditionalEncoder  # noqa: I900

if __name__ == "__main__":
    cfg = OmegaConf.structured(cfg)
    print("Configs:\n", OmegaConf.to_yaml(cfg), "", sep="*" * 50 + "\n")

    lol_dm = LOLDataModule(cfg["dataset"])
    lol_dm.setup()

    batch = next(iter(lol_dm.train_dataloader()))

    model = ConditionalEncoder(**cfg["model"])

    output = model(batch["image"])

    print("Outputs:")
    for k, v in output.items():
        print(k + ":", v.size())
