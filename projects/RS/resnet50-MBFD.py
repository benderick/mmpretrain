_base_ = [
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/datasets/my.py',
    '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/schedules/imagenet_bs256.py', '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/configs/_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_MBFD',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=45,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
