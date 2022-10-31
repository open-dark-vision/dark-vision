from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import read_image

from src.datasets.meta import PairedImageInput  # noqa: I900


class LOL(Dataset):
    def __init__(self, root: Path, train: bool = True):
        path = root / ("our485" if train else "eval15")

        self.images = sorted((path / "low/").glob("*.png"), key=lambda x: int(x.stem))
        self.targets = sorted((path / "high/").glob("*.png"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> PairedImageInput:
        image = read_image(str(self.images[index]))
        target = read_image(str(self.targets[index]))

        return PairedImageInput(image=image, target=target)


if __name__ == "__main__":
    ds = LOL(Path("data/LOL-fair"), train=True)
    print(ds[0])
    print(len(ds))
