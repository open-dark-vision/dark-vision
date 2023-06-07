from pathlib import Path
from torchvision.utils import save_image
import os


def divide_chunks(l: list, n: int):
     
    for i in range(0, len(l), n):
        yield l[i:i + n]

def save_predictions_decom(predictions: list, paths: list, mode: str = "train"):
    Path(f"images_DecomNet_{mode}/reflect_low").mkdir(parents=True, exist_ok=True)
    Path(f"images_DecomNet_{mode}/illum_low").mkdir(parents=True, exist_ok=True)
    Path(f"images_DecomNet_{mode}/reflect_high").mkdir(parents=True, exist_ok=True)
    Path(f"images_DecomNet_{mode}/illum_high").mkdir(parents=True, exist_ok=True)
    for i, p in zip(range(len(predictions)),paths):

        for j in range(predictions[i][0].shape[0]):

            filename = os.path.basename(p[j])

            save_image(
                predictions[i][0][j], f"images_DecomNet_{mode}/reflect_low/{filename}"
            )
            save_image(
                predictions[i][1][j], f"images_DecomNet_{mode}/illum_low/{filename}"
            )
            save_image(
                predictions[i][2][j], f"images_DecomNet_{mode}/reflect_high/{filename}"
            )
            save_image(
                predictions[i][3][j], f"images_DecomNet_{mode}/illum_high/{filename}"
            )


def save_predictions(predictions: list, paths: list, mode: str = "train"):
    Path(f"images_test_{mode}").mkdir(parents=True, exist_ok=True)

    for i, p in zip(range(len(predictions)),paths):
        
        for j in range(predictions[i].shape[0]):

            filename = os.path.basename(p[j])

            save_image(
                predictions[i][j], f"images_test_{mode}/{filename}"
            )