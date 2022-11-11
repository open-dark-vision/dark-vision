"""
Run it to make sure that configs support pair transforms.
"""
import argparse
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from src.datamodules import LOLDataModule  # noqa: I900


def plot_batch(images: torch.Tensor, targets: torch.Tensor):
    fig, axs = plt.subplots(2, images.size()[0], figsize=(13, 5))
    for i in range(images.size()[0]):
        axs[0][i].imshow(images[i].permute(1, 2, 0))
        axs[1][i].imshow(targets[i].permute(1, 2, 0))

        for j in range(2):
            axs[j][i].axis("off")

    fig.suptitle("Input / Ground truth")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/tests/lol_dataset_test.yaml"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print("Configs:\n", OmegaConf.to_yaml(conf), "", sep="*" * 50 + "\n")

    setup_timer_start = timer()
    lol_dm = LOLDataModule(conf["dataset"])
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

    # if target is not affected it has to be permuted like that (0, 3, 1, 2)
    plot_batch(batch["image"][:8], batch["target"][:8])
