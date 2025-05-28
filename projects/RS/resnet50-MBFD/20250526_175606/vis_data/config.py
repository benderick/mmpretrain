auto_scale_lr = dict(base_batch_size=256)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=45,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet_MBFD'),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=45,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        30,
        60,
        90,
    ], type='MultiStepLR')
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root=
        '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/data/NWPU-RESISC45-split/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=256,
        type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root=
        '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/data/NWPU-RESISC45-split/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                magnitude_level=9,
                magnitude_std=0.5,
                num_policies=2,
                policies='timm_increasing',
                total_level=10,
                type='RandAugment'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        magnitude_level=9,
        magnitude_std=0.5,
        num_policies=2,
        policies='timm_increasing',
        total_level=10,
        type='RandAugment'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root=
        '/icislab/volume3/benderick/futurama/openmmlab/mmpretrain/data/NWPU-RESISC45-split/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './resnet50-MBFD'
