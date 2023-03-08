import cv2
import matplotlib.pyplot as plt
import torch

from src.configs.experiments import bridimo_config as cfg  # noqa: I900
from src.models.BriDiMo import LitBriDiMo  # noqa: I900
from src.transforms import MCBFSTransform, PairedTransformForBriDiMo  # noqa: I900

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chk_path = "reproducibility/iogve6zm/checkpoints/bridimo-coco-013-loss-0.0020.ckpt"
# chk_path = "reproducibility/gosp12yc/checkpoints/bridimo-lol-011-psnr-22.6407.ckpt"
# chk_path = "reproducibility/40iralmr/checkpoints/bridimo-lol-017-loss-0.0040.ckpt"
model = LitBriDiMo.load_from_checkpoint(chk_path, config=cfg)
model.eval()
model.to(DEVICE)

dark = cv2.imread("data/LOL/eval15/low/1.png")[..., ::-1]
GT = cv2.imread("data/LOL/eval15/high/1.png")[..., ::-1]

transform = MCBFSTransform(alpha_const=25)
transform2 = PairedTransformForBriDiMo(test=True)

trans_out = transform(GT)
x_dimmed, GT2, x2_light, y2_light = (
    trans_out["image"],
    trans_out["target"],
    trans_out["source_lightness"],
    trans_out["target_lightness"],
)

trans2_out = transform2(dark, GT)
x, x_light, y_light = (
    trans2_out["image"],
    trans2_out["source_lightness"],
    trans2_out["target_lightness"],
)

# concatenate both inputs
x_input = torch.cat([x_dimmed.unsqueeze(0), x.unsqueeze(0)], dim=0)
x_light = torch.cat([x2_light.unsqueeze(0), x_light.unsqueeze(0)], dim=0)
y_light = torch.cat([y2_light.unsqueeze(0), y_light.unsqueeze(0) / 2], dim=0)

# move to device
x_input = x_input.to(DEVICE)
x_light = x_light.to(DEVICE)
y_light = y_light.to(DEVICE)
GT2 = GT2.to(DEVICE)

output = model(x_input, x_light, y_light)

# calculate psnr and ssim using torchmetrics
psnr_artificial = model.val_psnr(output[0].unsqueeze(0), GT2.unsqueeze(0))
ssim_artificial = model.val_ssim(output[0].unsqueeze(0), GT2.unsqueeze(0))

psnr_real = model.val_psnr(output[1].unsqueeze(0), GT2.unsqueeze(0))
ssim_real = model.val_ssim(output[1].unsqueeze(0), GT2.unsqueeze(0))

# plot the results with input and gt
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0, 0].imshow(dark)
ax[0, 0].set_title("Real dark")
ax[0, 1].imshow(GT)
ax[0, 1].set_title("GT")
ax[0, 2].imshow(output[1].detach().cpu().numpy().transpose(1, 2, 0))
ax[0, 2].set_title(f"real output psnr: {psnr_real:.2f} ssim: {ssim_real:.2f}")

ax[1, 0].imshow(x_dimmed[:3, :, :].numpy().transpose(1, 2, 0))
ax[1, 0].set_title("Artificial dark")
ax[1, 1].imshow(GT)
ax[1, 1].set_title("GT")
ax[1, 2].imshow(output[0].detach().cpu().numpy().transpose(1, 2, 0))
ax[1, 2].set_title(
    f"artificial output psnr: {psnr_artificial:.2f} ssim: {ssim_artificial:.2f}"
)

# remove the axis
for i in range(2):
    for j in range(3):
        ax[i, j].axis("off")


plt.tight_layout()
plt.savefig("comparison.png")
