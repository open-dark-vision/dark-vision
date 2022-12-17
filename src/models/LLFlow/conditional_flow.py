import torch
from torch import nn

from src.models.LLFlow.flow_layers import FlowStep, SqueezeLayer  # noqa: I900


class ConditionalFlow(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_levels=3,
        n_flow_steps=12,
        n_additional_steps=2,
        n_conditional_channels=64,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_levels = n_levels

        conditional_features_names = [f"feature_maps_{i+1}" for i in range(n_levels)]

        current_channels = in_channels
        for level in range(self.n_levels):
            current_channels = self.append_squeeze(current_channels)

            self.append_additional_flow_steps(n_additional_steps, current_channels)
            self.append_flow_steps(
                current_channels,
                n_flow_steps,
                conditional_features_names[level],
                n_conditional_channels,
            )

    def append_flow_steps(
        self, current_channels, n_steps, position, n_conditional_channels
    ):
        for _ in range(n_steps):
            self.layers.append(
                FlowStep(
                    in_channels=current_channels,
                    flow_coupling=True,
                    position=position,
                    conditional_channels=n_conditional_channels,
                )
            )

    def append_additional_flow_steps(self, n_steps, current_channels):
        for _ in range(n_steps):
            self.layers.append(
                FlowStep(in_channels=current_channels, flow_coupling=False)
            )

    def append_squeeze(self, current_channels):
        self.layers.append(SqueezeLayer(factor=2))
        return current_channels * 4

    def forward(
        self, z=None, gt=None, conditional_features=None, logdet=0.0, reverse=False
    ):
        return (
            self.encode(gt, conditional_features, logdet)
            if not reverse
            else self.decode(z, conditional_features, logdet)
        )

    def encode(self, gt, rrdbResults, logdet=0.0):
        features = gt

        for layer in self.layers:
            features, logdet = layer(
                features, logdet, reverse=False, conditional_features=rrdbResults
            )

        return features, logdet

    def decode(self, z, rrdbResults, logdet=0.0):
        for layer in reversed(self.layers):
            z, logdet = layer(
                z, logdet=logdet, reverse=True, conditional_features=rrdbResults
            )

        return z, logdet


if __name__ == "__main__":
    cond_features = {
        "feature_maps_1": torch.zeros((1, 64, 4, 4)),
        "feature_maps_2": torch.zeros((1, 64, 2, 2)),
        "feature_maps_3": torch.zeros((1, 64, 1, 1)),
    }
    images = torch.rand((1, 1, 8, 8))
    print(images)
    logdet = torch.zeros(1)

    flow = ConditionalFlow(in_channels=1)

    zs, logdet = flow(gt=images, conditional_features=cond_features, logdet=logdet)

    reconstructed, _ = flow(
        z=zs, conditional_features=cond_features, logdet=logdet, reverse=True
    )
    print()
    print(reconstructed)
