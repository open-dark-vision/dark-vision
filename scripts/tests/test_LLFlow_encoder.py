"""
Run it to make sure that your LLFlow conditional encoder configs are ok.
"""
import argparse

from omegaconf import OmegaConf

from src.datamodules import LOLDataModule  # noqa: I900
from src.models.LLFlow import ConditionalEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/tests/llflow_encoder.yaml"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print("Configs:\n", OmegaConf.to_yaml(conf), "", sep="*" * 50 + "\n")

    lol_dm = LOLDataModule(conf["dataset"])
    lol_dm.setup()

    batch = next(iter(lol_dm.train_dataloader()))

    model = ConditionalEncoder(**conf["model"]["encoder"])

    output = model(batch['image'])

    print("Outputs:")
    for k, v in output.items():
        print(k+':', v.size())
