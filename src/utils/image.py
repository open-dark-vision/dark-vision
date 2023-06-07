from pathlib import Path
from typing import Union

import cv2
import numpy as np


def read_image_cv2(path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """Read an image from a path."""
    print(path)
    image = cv2.imread(str(path))
    if grayscale:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
