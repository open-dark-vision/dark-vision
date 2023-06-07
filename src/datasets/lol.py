from pathlib import Path
from typing import List, Optional
import torch
import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split

from src.configs.base import LOLDatasetConfig  # noqa: I900
from src.datasets.meta import PairedImageInput, SixFoldImageInput  # noqa: I900
from src.transforms import load_transforms  # noqa: I900
from src.utils.image import read_image_cv2  # noqa: I900


class LOL(Dataset):
    def __init__(
        self,
        root: Path,
        indices: Optional[List[int]] = None,
        train: bool = True,
        pair_transform: Optional[A.Compose] = None,
        preload: bool = False,
        load_paths: bool = False,
    ):

        self.pair_transform = (
            pair_transform
            if pair_transform is not None
            else A.Compose(
                [ToTensorV2()],
                additional_targets={"target": "image"},
            )
        )
        
        path = root / ("our485" if train else "eval15")

        self.image_names = sorted(
            (path / "low/").glob("*.png"), key=lambda x: int(x.stem)
        )
        self.target_names = sorted(
            (path / "high/").glob("*.png"), key=lambda x: int(x.stem)
        )

        if indices is not None:
            self.image_names = [self.image_names[index] for index in indices]
            self.target_names = [self.target_names[index] for index in indices]

        self.loaded = preload
        if self.loaded:
            self.load_all_images()

        self.load_paths = load_paths
        if self.load_paths:
            self.load_all_paths()

        

    def load_all_images(self) -> None:
        self.loaded_images_ = []
        self.loaded_targets_ = []

        for index in range(len(self)):
            self.loaded_images_.append(read_image_cv2(self.image_names[index]))
            self.loaded_targets_.append(read_image_cv2(self.image_names[index]))

    def load_all_paths(self) -> None:
        self.images_paths = []
        self.targets_paths = []

        for index in range(len(self)):
            self.images_paths.append(self.image_names[index])
            self.targets_paths.append(self.target_names[index])

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> PairedImageInput:
        
        image = (
            self.loaded_images_[index]
            if self.loaded
            else read_image_cv2(self.image_names[index])
        )
        target = (
            self.loaded_targets_[index]
            if self.loaded
            else read_image_cv2(self.target_names[index])
        )
        
        transformed = self.pair_transform(image=image, target=target)
        image, target = transformed["image"], transformed["target"]

        return PairedImageInput(image=image, target=target)


class LOLDataModule(pl.LightningDataModule):
    def __init__(self, config: LOLDatasetConfig):
        super().__init__()
        self.root = Path(config.path)
        self.config = config


        self.train_transform, self.test_transform = load_transforms(config.transform)

    def setup(self, stage: Optional[str] = None):
        n_train_images = len(list((self.root / "our485/low/").glob("*.png")))
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(
            range(n_train_images), [1 - self.config.val_size, self.config.val_size], generator
        )

        
        self.train_ds = LOL(
            self.root,
            indices=train_indices,
            train=True,
            pair_transform=self.train_transform,
            preload=self.config.preload,
            load_paths= self.config.load_paths,
        )


        self.images_names_train = self.train_ds.images_paths
        self.targets_names_train = self.train_ds.targets_paths
        
        
        self.val_ds = LOL(
            self.root,
            indices=val_indices,
            train=True,
            pair_transform=self.test_transform,
            preload=self.config.preload,
            load_paths= self.config.load_paths,
        )

        self.images_names_val = self.val_ds.images_paths
        self.targets_names_val = self.val_ds.targets_paths
        
        self.test_ds = LOL(
            self.root,
            train=False,
            pair_transform=self.test_transform,
            preload=self.config.preload,
            load_paths= self.config.load_paths,
        )

        self.images_names_test = self.test_ds.images_paths
        self.targets_names_test = self.test_ds.targets_paths

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self):
        if self.config.predict_on_train:
            return self.train_dataloader()
        elif self.config.predict_on_val:
            return self.val_dataloader()
        else:
            return self.test_dataloader()

class LOL_DECOM(LOL):
    def __init__(
        self,
        root_lol: Path,
        root_decom: Path,
        indices: Optional[List[int]] = None,
        train: bool = True,
        six_transform: Optional[A.Compose] = None,
        preload: bool = False,
        load_paths: bool = False,
    ):

    
        self.six_transform = (
            six_transform
            if six_transform is not None
            else A.Compose(
                [ToTensorV2()],
                additional_targets={"target": "image"},
            )
        )
        
        path_lol = root_lol / ("our485" if train else "eval15")

        self.image_names = sorted(
            (path_lol / "low/").glob("*.png"), key=lambda x: int(x.stem)
        )
        self.target_names = sorted(
            (path_lol / "high/").glob("*.png"), key=lambda x: int(x.stem)
        )

        path_decom = root_decom / ("images_DecomNet_train" if train else "images_DecomNet_test")

        self.reflect_high_names = sorted(
            (path_decom / "reflect_high/").glob("*.png"), key=lambda x: int(x.stem)
        )
        self.reflect_low_names = sorted(
            (path_decom / "reflect_low/").glob("*.png"), key=lambda x: int(x.stem)
        )
        self.illum_high_names = sorted(
            (path_decom / "illum_high/").glob("*.png"), key=lambda x: int(x.stem)
        )
        self.illum_low_names = sorted(
            (path_decom / "illum_low/").glob("*.png"), key=lambda x: int(x.stem)
        )

        
        if indices is not None:
            self.image_names = [self.image_names[index] for index in indices]
            self.target_names = [self.target_names[index] for index in indices]
            self.reflect_high_names = [self.reflect_high_names[index] for index in indices]
            self.reflect_low_names = [self.reflect_low_names[index] for index in indices]
            self.illum_high_names = [self.illum_high_names[index] for index in indices]
            self.illum_low_names = [self.illum_low_names[index] for index in indices]

        self.loaded = preload
        if self.loaded:
            self.load_all_images()

        self.load_paths = load_paths
        if self.load_paths:
            self.load_all_paths()

    
    def load_all_images(self) -> None:
        self.loaded_images_ = []
        self.loaded_targets_ = []
        self.loaded_reflect_high_ = []
        self.loaded_reflect_low_ = []
        self.loaded_illum_high_ = []
        self.loaded_illum_low_ = []

        for index in range(len(self)):
            self.loaded_images_.append(read_image_cv2(self.image_names[index]))
            self.loaded_targets_.append(read_image_cv2(self.image_names[index]))
            self.loaded_reflect_high_.append(read_image_cv2(self.image_names[index]))
            self.loaded_reflect_low_.append(read_image_cv2(self.image_names[index]))
            self.loaded_illum_high_.append(read_image_cv2(self.image_names[index]))
            self.loaded_illum_low_.append(read_image_cv2(self.image_names[index]))

    def load_all_paths(self) -> None:
        self.images_paths = []
        self.targets_paths = []
        self.reflect_high_paths = []
        self.reflect_low_paths = []
        self.illum_high_paths = []
        self.illum_low_paths = []

        for index in range(len(self)):
            self.images_paths.append(self.image_names[index])
            self.targets_paths.append(self.target_names[index])
            self.reflect_high_paths.append(self.reflect_high_names[index])
            self.reflect_low_paths.append(self.reflect_low_names[index])
            self.illum_high_paths.append(self.illum_high_names[index])
            self.illum_low_paths.append(self.illum_low_names[index])

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> PairedImageInput:

        image = (
            self.loaded_images_[index]
            if self.loaded
            else read_image_cv2(self.image_names[index])
        )
        target = (
            self.loaded_targets_[index]
            if self.loaded
            else read_image_cv2(self.target_names[index])
        )

        reflect_high = (
            self.loaded_reflect_high_[index]
            if self.loaded
            else read_image_cv2(self.reflect_high_names[index])
        )

        reflect_low = (
            self.loaded_reflect_low_[index]
            if self.loaded
            else read_image_cv2(self.reflect_low_names[index])
        )

        illum_high = (
            self.loaded_illum_high_[index]
            if self.loaded
            else read_image_cv2(self.illum_high_names[index], grayscale=True)
        )

        illum_low = (
            self.loaded_illum_low_[index]
            if self.loaded
            else read_image_cv2(self.illum_low_names[index], grayscale=True)
        )

        transformed = self.six_transform(image=image, target=target, reflect_high=reflect_high, reflect_low=reflect_low, illum_high=illum_high, illum_low=illum_low)
        image, target, reflect_high, reflect_low, illum_high, illum_low = transformed["image"], transformed["target"], transformed["reflect_high"], transformed["reflect_low"], transformed["illum_high"], transformed["illum_low"]
        
        return SixFoldImageInput(image=image, target=target, reflect_high=reflect_high, reflect_low=reflect_low, illum_high=illum_high, illum_low=illum_low)  

class LOLDecompositionDataModul(LOLDataModule):
    def __init__(self, config: LOLDatasetConfig):

        LOLDataModule.__init__(self, config)
        self.root_lol = Path(config.path)
        self.root_decom = Path(config.path_decom)
        

    def setup(self, stage: Optional[str] = None):
        n_train_images = len(list((self.root / "our485/low/").glob("*.png")))
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(
            range(n_train_images), [1 - self.config.val_size, self.config.val_size], generator
        )

        
        self.train_ds = LOL_DECOM(
            self.root_lol,
            self.root_decom,
            indices=train_indices,
            train=True,
            six_transform=self.train_transform,
            preload=self.config.preload,
            load_paths= self.config.load_paths,
        )


        self.images_names_train = self.train_ds.images_paths
        self.targets_names_train = self.train_ds.targets_paths
        
        
        self.val_ds = LOL_DECOM(
            self.root_lol,
            self.root_decom,
            indices=val_indices,
            train=True,
            six_transform=self.test_transform,
            preload=self.config.preload,
            load_paths= self.config.load_paths,
        )

        self.images_names_val = self.val_ds.images_paths
        self.targets_names_val = self.val_ds.targets_paths
        
        self.test_ds = LOL_DECOM(
            self.root_lol,
            self.root_decom,
            train=False,
            six_transform=self.test_transform,
            preload=self.config.preload,
            load_paths= self.config.load_paths,
        )

        self.images_names_test = self.test_ds.images_paths
        self.targets_names_test = self.test_ds.targets_paths