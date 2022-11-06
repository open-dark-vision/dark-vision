import albumentations as A


class ImageEqualize(A.BasicTransform):
    def __init__(self, **kwargs):
        super().__init__(*kwargs)

    def __call__(self, **kwargs):
        kwargs["image"] = A.equalize(kwargs["image"])
        return kwargs
