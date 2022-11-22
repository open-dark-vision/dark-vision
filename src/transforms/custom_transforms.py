from typing import Dict, Tuple

import albumentations as A
import numpy as np
import torch


class LLFlowTransform:
    def __init__(self, train: bool = True, flip_prob: float = 0.5, crop_size: Tuple[int, int] = (160, 160)):
        self.train = train
        self.flip_prob = flip_prob
        self.crop_size = crop_size

    @staticmethod
    def gradient(image: np.array) -> Tuple[np.array, np.array]:
        def sub_gradient(x: np.array) -> np.array:
            left_shift_x, right_shift_x = np.zeros_like(x), np.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            return 0.5 * (left_shift_x - right_shift_x)

        return sub_gradient(image), sub_gradient(image.transpose(1, 0, 2)).transpose(1, 0, 2)

    @staticmethod
    def color_map(image: np.array) -> np.array:
        return image / (image.sum(axis=1, keepdims=True) + 1e-4)

    @staticmethod
    def noise_map(c_map: np.array) -> np.array:
        dx, dy = LLFlowTransform.gradient(c_map)
        return np.maximum(np.abs(dx), np.abs(dy))

    def common_horizontal_flip(self, image: np.array, target: np.array) -> Tuple[np.array, np.array]:
        if np.random.random() < self.flip_prob:
            return np.flip(image, 1), np.flip(target, 1)
        return image, target

    def common_random_crop(self, image: np.array, hist: np.array, target: np.array) -> Tuple[np.array, np.array, np.array]:
        width = image.shape[0]
        height = image.shape[1]

        start_x = np.random.randint(low=0, high=(width - self.crop_size[0]) + 1) if width > self.crop_size[0] else 0
        start_y = np.random.randint(low=0, high=(height - self.crop_size[1]) + 1) if height > self.crop_size[1] else 0

        crop_slice = np.s_[start_x:start_x + self.crop_size[0], start_y:start_y + self.crop_size[1], :]

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

        return {'image': torch.from_numpy(image).float(), 'target': torch.from_numpy(target).float()}
