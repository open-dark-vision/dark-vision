from src.configs.base import (  # noqa: I900
    DataSegmentationConfig,
    ExperimentConfig
)

segment_config = ExperimentConfig(
    name="segmentation_experiment",
    model_name='nvidia/mit-b0',
    dataset=DataSegmentationConfig(
        id2label_file=r".\id2label.json",
        dic_class={0: 'void', 1: 'road', 2: 'sidewalk', 3: 'building', 4: 'wall', 5: 'fence', 6: 'pole', 7: 'traffic light', 8: 'traffic sign', 9: 'vegetation', 10: 'terrain', 11: 'sky', 12: 'person', 13: 'rider', 14: 'car', 15: 'truck', 16: 'bus', 17: 'train', 18: 'motorcycle', 19: 'bicycle'},
        path=r".\nightcity_data",
        num_workers=0,
        pin_memory=False,
        transform=None, 
        batch_size=2,
        preload=False,
    ),
    device="cpu",
    epochs=20,
)