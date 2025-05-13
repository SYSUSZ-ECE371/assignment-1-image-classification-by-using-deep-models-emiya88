# Base configurations imported from MMPretrain
_base_ = [
    '_base_/models/resnet50.py',
    '_base_/datasets/imagenet_bs32.py',
    '_base_/schedules/imagenet_bs256.py',
    '_base_/default_runtime.py'
]

# Model configuration
model = dict(
    head=dict(
        num_classes=5,  # Number of flower dataset classes
        topk=(1,)       # Evaluate top-1 accuracy only
    )
)

# Path to pre-trained weights for fine-tuning
load_from = r'C:\Users\YiFeng\mmpretrain\data\resnet50_8xb32_in1k_20210831-ea4938fc.pth'

# Data preprocessing configuration
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],  # ImageNet mean for normalization
    std=[58.395, 57.12, 57.375],     # ImageNet std for normalization
    to_rgb=True,                     # Convert images to RGB
    num_classes=5,                   # Match dataset classes
)

# Dataset configuration
dataset_type = 'ImageNet' 
data_root = r'C:\Users\YiFeng\mmpretrain\data\flower_dataset'
classes = [c.strip() for c in open(f'{data_root}/classes.txt')] # Load class names

# Training dataloader configuration
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_prefix=f'{data_root}/train',
        ann_file=f'{data_root}/train.txt',
        # data_prefix='train',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
    ),
    # sampler=dict(shuffle=True, type='DefaultSampler'),
)

# Validation dataloader configuration
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_prefix=f'{data_root}/val',
        ann_file=f'{data_root}/val.txt',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
    ),
    # sampler=dict(shuffle=False, type='DefaultSampler'),
)

# Validation configuration and evaluator
val_cfg = dict()
val_evaluator = dict(type='Accuracy', topk=(1,))

# Optimizer configuration
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005,  
        momentum=0.9,
        weight_decay=0.0001,
    ),
)
# param_scheduler = dict(
#     type='MultiStepLR',
#     by_epoch=True,
#     milestones=[10, 15],
#     gamma=0.1,
# )

# Learning rate scaling
auto_scale_lr = dict(base_batch_size=256)

# Training configuration
train_cfg = dict(
    by_epoch=True,
    max_epochs=20,
    val_interval=1,
)

# Evaluation configuration
# val_cfg = dict()
# test_cfg = dict()

# load_from = 'checkpoints/resnet50_8xb32_in1k_20200831-f84a1f1f.pth'

# Working directory for outputs
# work_dir = 'work_dirs/resnet50_finetune_flower'