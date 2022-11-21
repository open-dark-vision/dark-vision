import argparse
from typing import Dict

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.configs.base import PairSelectionMethod  # noqa: I900
from src.configs.tests import (  # noqa: I900
    exdark_dataset_test_config,
    lol_dataset_test_config,
    sice_dataset_test_config,
    supplementary_dataset_test_config,
)
from src.datasets import (  # noqa: I900
    ExDarkDataModule,
    LOLDataModule,
    SICEDataModule,
    SupplementaryDataModule,
)

# Dataset: LOL
# means=[0.06179853901267052, 0.059038277715444565, 0.05392766743898392]
# stds=[0.06508711725473404, 0.06142551079392433, 0.06532308459281921]

# Dataset: EXDARK
# means=[0.20310050249099731, 0.15813758969306946, 0.12950506806373596]
# stds=[0.239516481757164, 0.20435328781604767, 0.19291265308856964]

# Dataset: SICE (HALF EXPOSURE)
# means=[0.3875998556613922, 0.3921565115451813, 0.3619844317436218]
# stds=[0.2814829647541046, 0.2902642488479614, 0.3096781075000763]


def load_datamodule(dataset_name: str) -> Dict:
    dataset_name = dataset_name.lower()
    if dataset_name == "lol":
        cfg = OmegaConf.structured(lol_dataset_test_config)
        dm = LOLDataModule(cfg.dataset)
    elif dataset_name == "exdark":
        cfg = OmegaConf.structured(exdark_dataset_test_config)
        dm = ExDarkDataModule(cfg.dataset)
    elif dataset_name == "sice":
        cfg = OmegaConf.structured(sice_dataset_test_config)
        cfg.dataset.train_pair_selection_method = PairSelectionMethod.HALFEXP_TARGET
        dm = SICEDataModule(cfg.dataset)
    elif dataset_name in ["dicm", "fusion", "lime", "low", "mef", "npe", "vv"]:
        cfg = OmegaConf.structured(supplementary_dataset_test_config)
        cfg.dataset.name = dataset_name.upper()
        dm = SupplementaryDataModule(cfg.dataset)
    else:
        raise ValueError(f"Dataset {dataset_name} not found in test configs.")

    return dm


def compute_means_and_stds(loader):
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    count = 0

    for batch in tqdm(loader):
        images = batch["image"]
        images = images.float() / 255.0

        for image in images:
            psum += image.sum(dim=(1, 2))
            psum_sq += (image**2).sum(dim=(1, 2))

            count += image.shape[1] * image.shape[2]

    total_mean = psum / count
    total_std = torch.sqrt((psum_sq / count) - (total_mean**2))

    return total_mean.numpy().tolist(), total_std.numpy().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean and std of a dataset.")
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset to compute the stats for.",
    )

    args = parser.parse_args()

    datamodule = load_datamodule(args.dataset)
    datamodule.setup()

    means, stds = compute_means_and_stds(datamodule.train_dataloader())

    print(f"Dataset: {args.dataset.upper()}")
    print(f"{means=}")
    print(f"{stds=}")
