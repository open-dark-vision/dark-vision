from src.configs.base import (  # noqa: I900
    BriDiMoFinetuneTransformConfig,
    BriDiMoModelConfig,
    BriDiMoTransformConfig,
    COCODatasetConfig,
    ExperimentConfig,
    LOLDatasetConfig,
    Loss,
    LossConfig,
    Optimizer,
    OptimizerConfig,
    Transform,
)

bridimo_config = ExperimentConfig(
    name="bridimo-coco",
    dataset=COCODatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=BriDiMoTransformConfig(
            name=Transform.MCBFS, image_size=256, test_alpha=25
        ),
        batch_size=2,
    ),
    model=BriDiMoModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAM, lr=5e-4, weight_decay=0, betas=(0.9, 0.99)
    ),
    loss=LossConfig(name=Loss.MSE),
    device="cpu",
    epochs=20,
)


bridimo_finetune_config = ExperimentConfig(
    name="bridimo-lol",
    dataset=LOLDatasetConfig(
        num_workers=0,
        pin_memory=False,
        transform=BriDiMoFinetuneTransformConfig(
            name=Transform.BDM_FINETUNE, image_size=256, flip_prob=0.5
        ),
        batch_size=16,
        preload=True,
    ),
    model=BriDiMoModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAM, lr=5e-4, weight_decay=0, betas=(0.9, 0.99)
    ),
    loss=LossConfig(name=Loss.MSE),
    device="gpu",
    epochs=50,
)
