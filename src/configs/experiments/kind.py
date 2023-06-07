
from src.configs.base import (  # noqa: I900
    ExperimentConfig,
    KinDLossConfig,
    KinDModelConfig,
    LOLDatasetConfig,
    Optimizer,
    Scheduler,
    OptimizerConfig,
    Transform,
    TransformConfig,
    SchedulerConfig
)

kind_config_decom = ExperimentConfig(
    name="kind-lol",
    model_name='KinD_decom',
    save_predictions=None,#'test',
    checkpoint_path=None,#r'.\lightning_logs\version_0\checkpoints\DecomNet\last.ckpt',
    dataset=LOLDatasetConfig(
        path=r".\data\LOL",
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(name=Transform.KIND, image_size=256), 
        batch_size=2,
        predict_on_train=False,
        predict_on_val=False,
        load_paths=True
    ),
    model=KinDModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAM, lr=1e-4, weight_decay=0, betas=(0.9, 0.99)
    ),
    loss=KinDLossConfig(),
    device="cpu",
    epochs=2000,
)


kind_config_illumin = ExperimentConfig(
    name="kind-lol",
    model_name='KinD_illumin',
    save_predictions=None,#'training',
    checkpoint_path = None,#r'.\lightning_logs\version_0\checkpoints\IlluminationAdjustNet\last.ckpt',
    dataset=LOLDatasetConfig(
        path=r".\data\LOL",
        path_decom=r".",
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(name=Transform.KIND_DECOM, image_size=256), 
        batch_size=2,
        predict_on_train=False,
        predict_on_val=False,
        load_paths=True
    ),
    model=KinDModelConfig(),
    optimizer=OptimizerConfig(
        name=Optimizer.ADAM, lr=1e-4, weight_decay=0, betas=(0.9, 0.99)
    ),
    loss=KinDLossConfig(),
    device="cpu",
    epochs=2000,
)


kind_config_restore = ExperimentConfig(
    name="kind-lol",
    model_name='KinD_restore',
    save_predictions=None,#'training',
    checkpoint_path= None,#r'.\lightning_logs\version_0\checkpoints\RestorationNet\last.ckpt',
    
    dataset=LOLDatasetConfig(
        path=r".\data\LOL",
        path_decom=r".\data",
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(name=Transform.KIND_DECOM, image_size=256), 
        batch_size=2,
        predict_on_train=False,
        predict_on_val=False,
        load_paths=True
    ),
    model=KinDModelConfig(),
    optimizer_restoration=OptimizerConfig(
        name=Optimizer.ADAM, lr=1e-4, weight_decay=0, scheduler=SchedulerConfig(name=Scheduler.KIND, frequency=1), betas=(0.9, 0.99)
    ),
    loss=KinDLossConfig(),
    device="cpu",
    epochs=2000,
)


kind_config_finale = ExperimentConfig(
    name="kind-lol",
    model_name='KinD_finale',
    save_predictions=None,#'training',
    checkpoint_path= None,#r'.\checkpoints_finale',
    dataset=LOLDatasetConfig(
        path=r".\data\LOL",
        num_workers=0,
        pin_memory=False,
        transform=TransformConfig(name=Transform.KIND, image_size=256), 
        batch_size=2,
        predict_on_train=False,
        predict_on_val=False,
        load_paths=True
    ),
    model=KinDModelConfig(),
    optimizer_restoration=OptimizerConfig(
        name=Optimizer.ADAM, lr=1e-4, weight_decay=0, scheduler=SchedulerConfig(name=Scheduler.KIND, frequency=1), betas=(0.9, 0.99)
    ),
    loss=KinDLossConfig(),
    device="cpu",
    epochs=2000,
)