from typing import Tuple

import albumentations as A
import numpy as np
import torch


class PairedTransformForBriDiMo:
    """
    Uses similar pipeline as MCBFSTransform but doesn't dim the photo.
    Instead takes already dark image with corresponding
    light one and augment them to fit BriDiMo
    """

    def __init__(self, flip_prob=0.5, crop_size=None, test=False):
        if crop_size is None:
            crop_size = 256

        self.flip_prob = flip_prob
        self.crop_size = (crop_size, crop_size)
        self.test = test

    def common_horizontal_flip(
        self, image: np.array, target: np.array
    ) -> Tuple[np.array, np.array]:
        if np.random.random() < self.flip_prob:
            return np.flip(image, 1), np.flip(target, 1)
        return image, target

    def common_random_crop(
        self, image: np.array, target: np.array
    ) -> Tuple[np.array, np.array]:
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

        return image[crop_slice], target[crop_slice]

    def __call__(self, image, target):
        """
        :param image: dark image (input)
        :param target: light image (ground truth)
        :return: dict of prepared dark image, target image and lightnesses of both
        """
        if not self.test:
            dark, light = self.common_horizontal_flip(image, target)
            dark, light = self.common_random_crop(dark, light)
        else:
            dark, light = image, target

        # get color map and luminance of image
        R, L = MCBFSTransform.retinex_decomposition(dark / 255.0)

        # get luminance of target and calc mean
        _, L_target = MCBFSTransform.retinex_decomposition(light / 255.0)

        # histogram equalization for image
        hist_eq = A.augmentations.functional.equalize(dark) / 255.0

        # concatenate and normalize all channels
        dark = np.concatenate([dark / 255.0, hist_eq, R], axis=2).transpose(2, 0, 1)

        dark = torch.from_numpy(dark).float()
        source_lightness = torch.tensor(L.mean()).float()
        target_lightness = torch.tensor(L_target.mean()).float()

        light = torch.tensor(light.transpose(2, 0, 1) / 255.0).float()

        return {
            "image": dark,
            "target": light,
            "source_lightness": source_lightness,
            "target_lightness": target_lightness,
        }


class MCBFSTransform:
    """Monte Carlo Bayer Filter Simulation Transform"""

    rng = np.random.default_rng()
    COLOR_P = 3
    FILTER_P = 4
    RETINEX = 3
    NORM_FACTOR = COLOR_P * FILTER_P * RETINEX

    def __init__(
        self,
        transforms=None,
        alpha_min=8,
        alpha_max=50,
        alpha_const=None,
        mc_max=30_000,
        gamma=2.2,
    ):
        """
        :param transforms: augmentation applied before MCBFS
        :param alpha_min: minimum value of randomized alpha
               parameter in Beta distribution
        :param alpha_max: maximum value of randomized alpha
               parameter in Beta distribution
        :param mc_max: maximum number of photons for white pixel
        :param gamma: gamma value (bigger than 0)
        """
        self.mc_max = mc_max
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_const = alpha_const
        self.transforms = transforms

    @staticmethod
    def retinex_decomposition(img):
        """
        Decompose an image into reflectance and luminance using the retinex theory
        :param img: input image
        :return: reflectance and luminance
        """
        R = img / (img.sum(axis=2, keepdims=True) + 1e-6)
        L = (img / (3 * R + 1e-6))[:, :, 0]
        return R, L

    def s_shape_transformation(self, L):
        """
        for gamma > 1 s-shape function for values between 0 and 1
        for gamma < 1 tan-shape function for values between 0 and 1
        gamma and 1/gamma are inverse of each other
        :param L: luminance values (between 0 and 1)
        :return: transformed luminance values
        """
        return 1.0 / (1 + ((L + 1e-6) / (1 - L)) ** -self.gamma)

    def inverse_s_shape_transformation(self, L):
        """
        for gamma > 1 s-shape function for values between 0 and 1
        for gamma < 1 tan-shape function for values between 0 and 1
        gamma and 1/gamma are inverse of each other
        :param L: luminance values (between 0 and 1)
        :return: transformed luminance values
        """
        return 1.0 / (1 + ((L + 1e-6) / (1 - L)) ** -(1.0 / self.gamma))

    def bayer_filter_mc(self, R, L):
        """
        Simulate photons occurrences and the bayer filter result
        :param R: reflectance values
        :param L: luminance values
        :return: simulated bayer filter result (deformed image)
        """
        # calculate the number of photons for each luminance value
        MC = (self.mc_max * L).astype(np.uint32)

        # simulate photons and the bayer filter result
        GT_photons = (
            np.random.binomial(
                MCBFSTransform.rng.multinomial(MC.flatten(), R.reshape(-1, 3)).astype(
                    np.int32
                ),
                [0.25, 0.5, 0.25],
            )
            .reshape(*R.shape)
            .astype(np.float32)
        )

        # normalize the green channel
        GT_photons[:, :, 1] /= 2.0

        # normalize the result and clip to [0, 1]
        return np.clip((MCBFSTransform.NORM_FACTOR * GT_photons) / self.mc_max, 0, 1)

    def pollute(self, img, alpha=10):
        """
        Dim the image and simulate the bayer filter result
        : img: input image
        : a: beta distribution parameter (between 0 and 100)
        : return: simulated bayer filter result (deformed image)
        """
        # decompose to retinex and luminance
        R, L_target = self.retinex_decomposition(img)

        # dim the luminance
        L = L_target * np.random.beta(alpha, 100 - alpha)

        # apply s-shape transformation
        L = self.s_shape_transformation(L)

        # perform bayer filter monte carlo simulation
        R, L = self.retinex_decomposition(self.bayer_filter_mc(R, L))

        # apply inverse s-shape transformation
        L_source = self.inverse_s_shape_transformation(L)

        # recombine
        return np.clip((R * L[:, :, None]) * 3, 0, 1), L_source.mean(), L_target.mean()

    def __call__(self, light, alpha=None):
        if alpha is None and self.alpha_const is None:
            alpha = np.random.randint(self.alpha_min, self.alpha_max)
        elif self.alpha_const is not None:
            alpha = self.alpha_const

        if self.transforms:
            light = self.transforms(image=light)["image"]

        dark, source_lightness, target_lightness = self.pollute(light / 255.0, alpha)
        dark = (255 * dark).astype(np.uint8)

        # histogram equalization
        hist = A.augmentations.functional.equalize(dark) / 255.0

        # color mapping
        c_map = dark / (dark.sum(axis=2, keepdims=True) + 1e-4)

        # normalize
        dark = dark / 255.0
        light = light / 255.0

        # concatenate all images to a single tensor
        dark = np.concatenate([dark, hist, c_map], axis=2)

        dark = dark.transpose(2, 0, 1)
        light = light.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(dark).float(),
            "target": torch.from_numpy(light).float(),
            "source_lightness": torch.tensor(source_lightness).float(),
            "target_lightness": torch.tensor(target_lightness).float(),
        }
