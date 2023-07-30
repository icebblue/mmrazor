_base_ = [
    'mmrazor::_base_/datasets/mmcls/cifar100_bs128.py',
    'mmrazor::_base_/schedules/mmcls/cifar100_bs256.py',
    'mmrazor::_base_/default_runtime.py'
]

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        # convert image from BGR to RGB
        bgr_to_rgb=False),
    architecture=dict(
        cfg_path=  # noqa: E251
        'mmrazor::vanilla/mmcls/vit/vit-small_2xb128_cifar100.py',
        pretrained=False),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmrazor::vanilla/mmcls/resnet/resnet50_2xb128_cifar100.py',
        pretrained=False),
    teacher_ckpt=  # noqa: E251
    '/mnt/cephfs/home/alvin/yangzehang/workplace/DeRy/work_dirs/resnet50_2xb128_cifar100_adamw/epoch_200.pth',  # noqa: E501
    calculate_student_loss=True,
    student_trainable=True,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            # bb_1=dict(type='ModuleOutputs', source='backbone.timm_model.blocks.1'),
            bb_2=dict(type='ModuleOutputs', source='backbone.timm_model.blocks.8'),
            ),
        teacher_recorders=dict(
            # bb_1=dict(type='ModuleOutputs', source='backbone.layer2.1'),
            bb_2=dict(type='ModuleOutputs', source='backbone.layer3.4'),
            ),
        distill_losses=dict(
            # loss_1=dict(type='L2Loss', loss_weight=10),
            loss_2=dict(type='L2Loss', loss_weight=10),
            ),
        connectors=dict(
            # loss_1_sfeat=dict(
            #     type='Tran2ConvConnector',
            #     in_channel=384,
            #     out_channel=512,
            #     hw=(14,14),
            #     stride=2,
            #     ),
            loss_2_sfeat=dict(
                type='Tran2ConvConnector',
                in_channel=384,
                out_channel=1024,
                hw=(14,14),
                stride=1,
                ),
            ),
        loss_forward_mappings=dict(
            # loss_1=dict(
            #     s_feature=dict(
            #         from_student=True,
            #         recorder='bb_1',
            #         # record_idx=1,
            #         connector='loss_1_sfeat'),
            #     t_feature=dict(
            #         from_student=False,
            #         recorder='bb_1',
            #         # record_idx=2,
            #         ),
            # ),
            loss_2=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_2',
                    # record_idx=1,
                    connector='loss_2_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='bb_2',
                    # record_idx=2,
                    ),
            ),
        )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')