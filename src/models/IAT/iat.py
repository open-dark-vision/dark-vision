import torch
from einops import einsum, rearrange
from torch import nn

from src.models.IAT.global_net import GlobalNet  # noqa: I900
from src.models.IAT.local_net import LocalNet  # noqa: I900

# based on https://github.com/cuiziteng/Illumination-Adaptive-Transformer


class IAT(nn.Module):
    def __init__(self, in_dim=3, task_type="lol", layers_type="ccc"):
        super().__init__()

        self.local_net = LocalNet(in_dim=in_dim, layers_type=layers_type)
        self.global_net = GlobalNet(in_channels=in_dim, task_type=task_type)

    def apply_color(self, image, ccm):
        image = einsum(image, ccm, "h w c, k c -> h w k")
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        mul, add = self.local_net(img_low)
        img_high = img_low * mul + add

        gamma, color = self.global_net(img_low)
        b = img_high.shape[0]
        img_high = rearrange(img_high, "b c h w -> b h w c")
        img_high = torch.stack(
            [
                self.apply_color(img_high[i, ...], color[i, ...]) ** gamma[i, :]
                for i in range(b)
            ],
            dim=0,
        )
        img_high = rearrange(img_high, "b h w c -> b c h w")

        return mul, add, img_high


if __name__ == "__main__":
    device = "mps"
    img = torch.Tensor(5, 3, 400, 600).to(device)
    net = IAT(layers_type="cct").to(device)
    print("total parameters:", sum(param.numel() for param in net.parameters()))
    _, _, high = net(img)
    print(high.shape)
