import albumentations as A
import numpy as np
import torch


class MCBFSTransform:
    """Monte Carlo Bayer Filter Simulation Transform"""

    rng = np.random.default_rng()
    COLOR_P = 3
    FILTER_P = 4
    RETINEX = 3
    NORM_FACTOR = COLOR_P * FILTER_P * RETINEX

    def __init__(
        self, transforms=None, alpha_min=8, alpha_max=60, mc_max=30_000, gamma=2.2
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
        if alpha is None:
            alpha = np.random.randint(self.alpha_min, self.alpha_max)

        if self.transforms:
            light = self.transforms(light)

        dark, source_lightness, target_lightness = self.pollute(light, alpha)
        dark = (255 * dark).astype(np.uint8)

        # histogram equalization
        hist = A.augmentations.functional.equalize(dark) / 255.0

        # color mapping
        c_map = dark / (dark.sum(axis=2, keepdims=True) + 1e-4)

        # normalize
        dark = dark / 255.0

        # concatenate all images to a single tensor
        dark = np.concatenate([dark, hist, c_map], axis=2)

        dark = dark.transpose(2, 0, 1)
        light = light.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(dark).float(),
            "target": torch.from_numpy(light).float(),
            "source_lightness": torch.from_numpy(source_lightness).float(),
            "target_lightness": torch.from_numpy(target_lightness).float(),
        }
