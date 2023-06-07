from typing import Dict, Tuple, Union
import cv2
import albumentations as A
import numpy as np
import torch


class LLFlowTransform:
    def __init__(
        self,
        train: bool = True,
        flip_prob: float = 0.5,
        crop_size: Union[int, Tuple[int, int]] = 160,
    ):
        self.train = train
        self.flip_prob = flip_prob
        self.crop_size = (
            crop_size if type(crop_size) == tuple else (crop_size, crop_size)
        )

    @staticmethod
    def gradient(image: np.array) -> Tuple[np.array, np.array]:
        def sub_gradient(x: np.array) -> np.array:
            left_shift_x, right_shift_x = np.zeros_like(x), np.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            return 0.5 * (left_shift_x - right_shift_x)

        return sub_gradient(image), sub_gradient(image.transpose(1, 0, 2)).transpose(
            1, 0, 2
        )

    @staticmethod
    def color_map(image: np.array) -> np.array:
        return image / (image.sum(axis=1, keepdims=True) + 1e-4)

    @staticmethod
    def noise_map(c_map: np.array) -> np.array:
        dx, dy = LLFlowTransform.gradient(c_map)
        return np.maximum(np.abs(dx), np.abs(dy))

    def common_horizontal_flip(
        self, image: np.array, target: np.array
    ) -> Tuple[np.array, np.array]:
        if np.random.random() < self.flip_prob:
            return np.flip(image, 1), np.flip(target, 1)
        return image, target

    def common_random_crop(
        self, image: np.array, hist: np.array, target: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        width = image.shape[0]
        height = image.shape[1]

        start_x = (
            np.random.randint(low=0, high=(width - self.crop_size[0]) + 1)
            if width > self.crop_size[0]
            else 0
        )
        start_y = (
            np.random.randint(low=0, high=(height - self.crop_size[1]) + 1)
            if height > self.crop_size[1]
            else 0
        )

        crop_slice = np.s_[
            start_x : start_x + self.crop_size[0],
            start_y : start_y + self.crop_size[1],
            :,
        ]

        return image[crop_slice], hist[crop_slice], target[crop_slice]

    def __call__(self, image: np.array, target: np.array) -> Dict[str, torch.Tensor]:
        if self.train:
            image, target = self.common_horizontal_flip(image, target)

        hist = A.augmentations.functional.equalize(image)

        if self.train:
            image, hist, target = self.common_random_crop(image, hist, target)

        c_map = LLFlowTransform.color_map(image)
        n_map = LLFlowTransform.noise_map(c_map)

        image = image / 255.0
        hist = hist / 255.0
        target = target / 255.0

        image = np.log(np.clip(image + 1e-3, a_min=1e-3, a_max=1))

        image = np.concatenate([image, hist, c_map, n_map], axis=2)

        image = image.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(target).float(),
        }
    
class KinDTransform:
    def __init__(
        self,
        train: bool = True,
        mode: int = 0,
        crop_size: Union[int, Tuple[int, int]] = 48
    ):
        self.train = train
        self.mode = mode
        self.crop_size = (
            crop_size if type(crop_size) == tuple else (crop_size, crop_size))

    def random_crop(self, image: np.array, target: np.array
    ) -> Tuple[np.array, np.array]:
        width = image.shape[0]
        height = image.shape[1]
        
        start_x = (
            np.random.randint(low=0, high=width - self.crop_size[0])
            if width > self.crop_size[0]
            else 0
        )
        start_y = (
            np.random.randint(low=0, high=height - self.crop_size[1])
            if height > self.crop_size[1]
            else 0
        )

        crop_slice = np.s_[
            start_x : start_x + self.crop_size[0],
            start_y : start_y + self.crop_size[1],
            :,
        ]

        return image[crop_slice], target[crop_slice]
        

    def data_augmentation(self, image: np.array) -> np.array:
        if self.mode == 0:
            # original
            return image
        elif self.mode == 1:
            # flip up and down
            return np.flipud(image)
        elif self.mode == 2:
            # rotate counterwise 90 degree
            return np.rot90(image)
        elif self.mode == 3:
            # rotate 90 degree and flip up and down
            image = np.rot90(image)
            return np.flipud(image)
        elif self.mode == 4:
            # rotate 180 degree
            return np.rot90(image, k=2)
        elif self.mode == 5:
            # rotate 180 degree and flip
            image = np.rot90(image, k=2)
            return np.flipud(image)
        elif self.mode == 6:
            # rotate 270 degree
            return np.rot90(image, k=3)
        elif self.mode == 7:
            # rotate 270 degree and flip
            image = np.rot90(image, k=3)
            return np.flipud(image)
        
    def __call__(self, image: np.array, target: np.array) -> Dict[str, torch.Tensor]:
        
        image = cv2.resize(image, self.crop_size, interpolation = cv2.INTER_AREA)
        target = cv2.resize(target, self.crop_size, interpolation = cv2.INTER_AREA)

        if self.train:
            image, target = self.random_crop(image, target)

        if self.train:
            image, target = self.data_augmentation(image), self.data_augmentation(target)

        image = np.array(image, dtype="float32")  / 255.0
        target = np.array(target, dtype="float32")  / 255.0
    
        image_norm = np.float32((image - np.min(image)) / np.maximum((np.max(image) - np.min(image)), 0.001))
        target_norm = np.float32((target - np.min(target)) / np.maximum((np.max(target) - np.min(target)), 0.001))
    

        return {
            "image": torch.from_numpy(image_norm).float(),
            "target": torch.from_numpy(target_norm).float(),
        }

class KinDTransform_DECOM:
    def __init__(
        self,
        train: bool = True,
        mode: int = 0,
        crop_size: Union[int, Tuple[int, int]] = 48
    ):
        self.train = train
        self.mode = mode
        self.crop_size = (
            crop_size if type(crop_size) == tuple else (crop_size, crop_size))

    def random_crop(self, image: np.array, target: np.array, reflect_high: np.array, reflect_low: np.array, ill_high: np.array, ill_low: np.array
    ) -> Tuple[np.array, np.array]:
        width = image.shape[0]
        height = image.shape[1]
        
        start_x = (
            np.random.randint(low=0, high=width - self.crop_size[0])
            if width > self.crop_size[0]
            else 0
        )
        start_y = (
            np.random.randint(low=0, high=height - self.crop_size[1])
            if height > self.crop_size[1]
            else 0
        )

        crop_slice = np.s_[
            start_x : start_x + self.crop_size[0],
            start_y : start_y + self.crop_size[1],
            :,
        ]
       
        return image[crop_slice], target[crop_slice], reflect_high[crop_slice], reflect_low[crop_slice], ill_high[crop_slice], ill_low[crop_slice]
        

    def data_augmentation(self, image: np.array) -> np.array:
        if self.mode == 0:
            # original
            return image
        elif self.mode == 1:
            # flip up and down
            return np.flipud(image)
        elif self.mode == 2:
            # rotate counterwise 90 degree
            return np.rot90(image)
        elif self.mode == 3:
            # rotate 90 degree and flip up and down
            image = np.rot90(image)
            return np.flipud(image)
        elif self.mode == 4:
            # rotate 180 degree
            return np.rot90(image, k=2)
        elif self.mode == 5:
            # rotate 180 degree and flip
            image = np.rot90(image, k=2)
            return np.flipud(image)
        elif self.mode == 6:
            # rotate 270 degree
            return np.rot90(image, k=3)
        elif self.mode == 7:
            # rotate 270 degree and flip
            image = np.rot90(image, k=3)
            return np.flipud(image)
        
    def __call__(self, image: np.array, target: np.array, reflect_high: np.array, reflect_low: np.array, illum_high: np.array, illum_low: np.array
        ) -> Dict[str, torch.Tensor]:
        
        image = cv2.resize(image, self.crop_size, interpolation = cv2.INTER_AREA)
        target = cv2.resize(target, self.crop_size, interpolation = cv2.INTER_AREA)
        reflect_high = cv2.resize(reflect_high, self.crop_size, interpolation = cv2.INTER_AREA)
        reflect_low = cv2.resize(reflect_low, self.crop_size, interpolation = cv2.INTER_AREA)
        illum_high = cv2.resize(illum_high, self.crop_size, interpolation = cv2.INTER_AREA)
        illum_low = cv2.resize(illum_low, self.crop_size, interpolation = cv2.INTER_AREA)
        illum_low = np.expand_dims(illum_low, axis=2)
        illum_high = np.expand_dims(illum_high, axis=2)
        
        if self.train:
        
            image, target, reflect_high, reflect_low, illum_high, illum_low = self.random_crop(image, target, reflect_high, reflect_low, illum_high, illum_low)
            
        if self.train:
            image, target, reflect_high, reflect_low, illum_high, illum_low = self.data_augmentation(image), self.data_augmentation(target), self.data_augmentation(reflect_high), self.data_augmentation(reflect_low), self.data_augmentation(illum_high), self.data_augmentation(illum_low)

        image = np.array(image, dtype="float32")  / 255.0
        target = np.array(target, dtype="float32")  / 255.0
        reflect_high = np.array(reflect_high, dtype="float32")  / 255.0
        reflect_low = np.array(reflect_low, dtype="float32")  / 255.0
        illum_high = np.array(illum_high, dtype="float32")  / 255.0
        illum_low = np.array(illum_low, dtype="float32")  / 255.0
    
        image_norm = np.float32((image - np.min(image)) / np.maximum((np.max(image) - np.min(image)), 0.001))
        target_norm = np.float32((target - np.min(target)) / np.maximum((np.max(target) - np.min(target)), 0.001))
        
        
        return {
            "image": torch.from_numpy(image_norm).float(),
            "target": torch.from_numpy(target_norm).float(),
            "reflect_high": torch.from_numpy(reflect_high).float(),
            "reflect_low": torch.from_numpy(reflect_low).float(),
            "illum_high": torch.from_numpy(illum_high).float(),
            "illum_low": torch.from_numpy(illum_low).float(),
        }

    