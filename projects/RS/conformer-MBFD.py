_base_ = [
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/datasets/my.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/default_runtime.py'
]


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Conformer_MBFD', arch='tiny', drop_path_rate=0.1, init_cfg=None),
    neck=None,
    head=dict(
        type='ConformerHead',
        num_classes=45,
        in_channels=[256, 384],
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)


train_cfg = dict(max_epochs=100)
