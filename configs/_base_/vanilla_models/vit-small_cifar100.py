model = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(
        _scope_='mmrazor',
        type='TIMMBackbone',
        model_name='vit_small_patch16_224',
        pretrained=True),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=384,
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