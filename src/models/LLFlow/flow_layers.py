from typing import Optional, Tuple

import torch
from torch import nn

import src.models.LLFlow.utils as utils  # noqa: I900


class FlowLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.tensor,
        logdet: Optional[float] = None,
        reverse: bool = False,
        condition_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.tensor, float]:
        pass


class SqueezeLayer(FlowLayer):
    """
    Reduces the height and width of the image by the factor
    (typically factor=2) and increase number of
    channels by factor*factor. Selects pixels
    with dilation equals to factor value.
    """

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def squeeze(self, x):
        if self.factor == 1:
            return x

        B, C, H, W = x.size()

        assert (
            H % self.factor == 0 and W % self.factor == 0
        ), "Image shape has to be even. Got {}".format((H, W, self.factor))

        x = x.view(B, C, H // self.factor, self.factor, W // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(
            B, C * self.factor * self.factor, H // self.factor, W // self.factor
        )

    def unsqueeze(self, x):
        if self.factor == 1:
            return x

        factor2 = self.factor**2

        B, C, H, W = x.size()
        assert C % factor2 == 0, "{}".format(C)

        x = x.view(B, C // factor2, self.factor, self.factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(B, C // factor2, H * self.factor, W * self.factor)

    def forward(self, x, logdet=None, reverse=False, conditional_features=None):
        return self.squeeze(x) if not reverse else self.unsqueeze(x), logdet


class ActNorm2d(FlowLayer):
    """
    Approximates the mean and deviation of each channel
    and normalizes each batch using the calculated
    statistics (which can change in the learning process).
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(1, num_features, 1, 1))
        )
        self.register_parameter(
            "logs", nn.Parameter(torch.zeros(1, num_features, 1, 1))
        )
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def initialize_parameters(self, x):
        if not self.training:
            return

        if (self.bias != 0).any():
            self.inited = True
            return

        assert (
            x.device == self.bias.device
        ), f"Input and module are on different device ({x.device}, {self.bias.device})"

        with torch.no_grad():
            bias = -x.mean(dim=(0, 2, 3), keepdim=True)
            variation = (x + bias).square().mean(dim=(0, 2, 3), keepdim=True)

            logs = torch.log(self.scale / (torch.sqrt(variation) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

        self.inited = True

    def _center(self, x, reverse=False):
        return x + self.bias if not reverse else x - self.bias

    def _scale(self, x, logdet=None, reverse=False):
        x *= torch.exp(self.logs) if not reverse else torch.exp(-self.logs)

        if logdet is not None:
            dlogdet = self.logs.sum() * (x.size(2) * x.size(3))
            logdet += dlogdet if not reverse else -dlogdet

        return x, logdet

    def forward(self, x, logdet=None, reverse=False, conditional_features=None):
        if not self.inited:
            self.initialize_parameters(x)

        if not reverse:
            x = self._center(x, reverse)
            x, logdet = self._scale(x, logdet, reverse)
        else:
            x, logdet = self._scale(x, logdet, reverse)
            x = self._center(x, reverse)

        return (x, logdet) if logdet is not None else x


class InvertibleConv1x1(FlowLayer):
    """
    Uses QR decomposition on random square matrix to
    find a random orthogonal matrix as a weight of conv layer.
    That will guarantee that the matrix has an inverse (equals to transposition)
    Note that during the learning process it will no longer stay orthogonal.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        w_init = torch.linalg.qr(torch.rand(n_channels, n_channels))[0].float()
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))

    def get_weight(self, x, reverse):
        pixels = x.size(2) * x.size(3)
        dlogdet = torch.slogdet(self.weight)[1] * pixels

        # should we add condition if the matrix is invertible? (usually works without)
        weight = self.weight if not reverse else self.weight.inverse()

        return weight.view(self.n_channels, self.n_channels, 1, 1), dlogdet

    def forward(self, x, logdet=None, reverse=False, conditional_features=None):
        weight, dlogdet = self.get_weight(x, reverse)

        z = nn.functional.conv2d(x, weight)

        if logdet is not None:
            logdet += dlogdet if not reverse else -dlogdet

        return z, logdet


class ConditionalCouplingLayer(FlowLayer):
    AFFINE_EPS = 1e-4

    def __init__(
        self,
        in_channels,
        in_channels_rrdb=64,
        n_hidden_layers=1,
        hidden_channels=64,
        kernel_hidden=(1, 1),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_hidden = kernel_hidden
        self.n_hidden_layers = n_hidden_layers
        self.hidden_channels = hidden_channels

        self.channels_for_nn = in_channels // 2
        self.channels_for_co = in_channels - self.channels_for_nn

        self.affine_net = self.create_sequential(
            in_channels=self.channels_for_nn + in_channels_rrdb,
            out_channels=self.channels_for_co * 2,
        )

        self.features_net = self.create_sequential(
            in_channels=in_channels_rrdb, out_channels=in_channels * 2
        )

    def encode(self, z: torch.Tensor, logdet=None, ft=None):
        assert (
            z.shape[1] == self.in_channels
        ), f"Expected {self.in_channels} channels. Got {z.shape[1]} instead."

        # Feature Conditional
        out_z = self.features_net(ft)
        scaleFt, shiftFt = self.feature_extract(out_z)
        z = (z + shiftFt) * scaleFt
        logdet += scaleFt.log().sum(dim=(1, 2, 3))

        # Self Conditional
        z1, z2 = self.split(z)
        out_z = self.affine_net(torch.cat([z1, ft], dim=1))
        scale, shift = self.feature_extract(out_z)

        self.asserts(scale, shift, z1, z2)

        z2 = (z2 + shift) * scale
        logdet += scale.log().sum(dim=(1, 2, 3))

        z = torch.cat([z1, z2], dim=1)

        return z, logdet

    def decode(self, z: torch.Tensor, logdet=None, ft=None):
        # Self Conditional
        z1, z2 = self.split(z)
        out_z = self.affine_net(torch.cat([z1, ft], dim=1))
        scale, shift = self.feature_extract(out_z)

        self.asserts(scale, shift, z1, z2)

        z2 = (z2 / scale) - shift
        z = torch.cat([z1, z2], dim=1)

        logdet -= scale.log().sum(dim=(1, 2, 3))

        # Feature Conditional
        out_z = self.features_net(ft)
        scaleFt, shiftFt = self.feature_extract(out_z)
        z = (z / scaleFt) - shiftFt

        logdet -= scaleFt.log().sum(dim=(1, 2, 3))

        return z, logdet

    def forward(
        self, x: torch.Tensor, logdet=None, reverse=False, conditional_features=None
    ):
        return (
            self.encode(x, logdet, conditional_features)
            if not reverse
            else self.decode(x, logdet, conditional_features)
        )

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (
            f"Expected z1 channels to be the same as "
            f"channels for nn. Got {(z1.shape[1], self.channels_for_nn)}"
        )
        assert z2.shape[1] == self.channels_for_co, (
            f"Expected z2 channels to be the same as channels for coupling layer. "
            f"Got {(z2.shape[1], self.channels_for_co)}"
        )
        assert scale.shape[1] == shift.shape[1], (
            f"Expected scale and shift number of "
            f"channels to be the same. Got {(scale.shape[1], shift.shape[1])}"
        )
        assert scale.shape[1] == z2.shape[1], (
            f"Expected scale and z2 to have the same "
            f"number of channel. Got {(scale.shape[1], z2.shape[1])}"
        )

    @staticmethod
    def feature_extract(z):
        shift, scale = utils.split_channels(z)
        scale = torch.sigmoid(scale + 2.0) + ConditionalCouplingLayer.AFFINE_EPS
        return scale, shift

    def split(self, z):
        z1 = z[:, : self.channels_for_nn]
        z2 = z[:, self.channels_for_nn :]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (
            z1.shape[1],
            z2.shape[1],
            z.shape[1],
        )
        return z1, z2

    def create_sequential(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=False),
        ]

        for _ in range(self.n_hidden_layers):
            layers.append(
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=self.kernel_hidden,
                )
            )
            layers.append(ActNorm2d(self.hidden_channels))
            layers.append(nn.ReLU(inplace=False))

        layers.append(utils.Conv2dZeros(self.hidden_channels, out_channels, padding=1))

        return nn.Sequential(*layers)


class FlowStep(FlowLayer):
    def __init__(
        self, in_channels, flow_coupling=True, position=None, conditional_channels=64
    ):
        super().__init__()
        self.actnorm = ActNorm2d(in_channels)
        self.flow_permutation = InvertibleConv1x1(in_channels)
        self.coupling_layer = (
            ConditionalCouplingLayer(
                in_channels=in_channels, in_channels_rrdb=conditional_channels
            )
            if flow_coupling
            else None
        )

        self.position = position

    def forward(self, x, logdet=None, reverse=False, conditional_features=None):
        return (
            self.normal_flow(x, logdet, conditional_features)
            if not reverse
            else self.reverse_flow(x, logdet, conditional_features)
        )

    def normal_flow(self, z, logdet, cond_features=None):
        z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        z, logdet = self.flow_permutation(z, logdet, reverse=False)

        if self.coupling_layer:
            img_ft = cond_features[self.position]
            z, logdet = self.coupling_layer(
                z, logdet=logdet, reverse=False, conditional_features=img_ft
            )

        return z, logdet

    def reverse_flow(self, z, logdet, cond_features=None):
        if self.coupling_layer:
            img_ft = cond_features[self.position]
            z, logdet = self.coupling_layer(
                z, logdet=logdet, reverse=True, conditional_features=img_ft
            )

        z, logdet = self.flow_permutation(z, logdet, reverse=True)

        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet
