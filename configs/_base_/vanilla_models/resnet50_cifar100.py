model = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(
        _scope_='mmrazor',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=100,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5)),
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.1),
    #     dict(type='CutMix', alpha=1.0)
    # ])
)

find_unused_parameters = True