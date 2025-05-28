_base_ = [
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/datasets/my.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/schedules/imagenet_bs2048_AdamW.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/default_runtime.py'
]

train_cfg = dict(max_epochs=100, val_interval=5)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='MViT_MBFD', arch='base', drop_path_rate=0.1),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        in_channels=768,
        num_classes=45,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))


# dataset settings
train_dataloader = dict(batch_size=64)
val_dataloader = dict(batch_size=64)
test_dataloader = dict(batch_size=64)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(lr=2.5e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.pos_embed': dict(decay_mult=0.0),
            '.rel_pos_h': dict(decay_mult=0.0),
            '.rel_pos_w': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=1.0),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=30,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=30)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
