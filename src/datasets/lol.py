from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import read_image

from src.datasets.meta import PairedImageInput  # noqa: I900


class LOL(Dataset):
    def __init__(self, root: Path):
        self.images = sorted((root / "low/").glob("*.png"), key=lambda x: int(x.stem))

        self.targets = sorted((root / "high/").glob("*.png"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> PairedImageInput:
        image = read_image(str(self.images[index]))
        target = read_image(str(self.targets[index]))

        return PairedImageInput(image=image, target=target)


if __name__ == "__main__":
    ds = LOL(Path("data/LOL-fair/our485"))
    print(ds[0])
    print(len(ds))
