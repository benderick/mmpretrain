_base_ = [
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/datasets/my.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/default_runtime.py',
]

train_cfg = dict(max_epochs=100)

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt_MBFD',
        arch='tiny',
        drop_path_rate=0.2,
        layer_scale_init_value=0.,
        use_grn=True,
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=45,
        in_channels=768,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.2),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)


# dataset setting
train_dataloader = dict(batch_size=64)
val_dataloader = dict(batch_size=64)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=3.2e-3),
    clip_grad=None,
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=40,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=40)
]


# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]
