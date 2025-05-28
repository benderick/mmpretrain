_base_ = [
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/datasets/my.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/default_runtime.py'
]

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvNeXt_MBFD', arch='tiny', drop_path_rate=0.1),
    head=dict(
        type='LinearClsHead',
        num_classes=45,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)

# ---

# dataset setting
train_dataloader = dict(batch_size=64)
val_dataloader = dict(batch_size=64)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=None,
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)

