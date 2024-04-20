_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/inception_i3d/top-i3d_from-scratch_4xb16-16x4x1-100e_ucf101-rgb.py']

student = _base_.model
teacher_ckpt = '/mnt/cephfs/dataset/m3lab_data-z/pengying/checkpoints/mmaction2/swin-tiny-p244-w877_k400-pre_4xb4-16x4x1-50e_ucf101-rgb/best_acc_top1_epoch_49.pth'

data_preprocessor=dict(
    type='mmaction.ActionDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    format_shape='NCTHW')

model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='HetTeacherDistill',
    data_preprocessor=data_preprocessor,
    architecture=student,
    teacher=dict(
        cfg_path='mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/swin/swin-tiny-p244-w877_k400-pre_4xb4-16x4x1-50e_ucf101-rgb.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    frames_downsample_rate=4,
    is_teacher_downsample=True,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.Conv3d_2c_3x3'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.Mixed_3c'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.Mixed_4f'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.Mixed_5c'),),
        teacher_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layers.0.blocks.1.mlp'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layers.1.blocks.1.mlp'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layers.2.blocks.5.mlp'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.layers.3.blocks.1.mlp')),
        distill_losses=dict(
            at_loss_s1=dict(type='ATLoss', loss_weight=100),
            at_loss_s2=dict(type='ATLoss', loss_weight=100),
            at_loss_s3=dict(type='ATLoss', loss_weight=100),
            at_loss_s4=dict(type='ATLoss', loss_weight=100)),
        connectors=dict(
            # at_lodd_sfeat=dict(type='atkdorthogonal'),
            at_loss_tfeat=dict(type='SwinFeatureProjector')),
        loss_forward_mappings=dict(
            at_loss_s1=dict(
                s_feature=dict(from_student=True, recorder='bb_s1'),
                t_feature=dict(from_student=False, recorder='bb_s1', connector='at_loss_tfeat')),
            at_loss_s2=dict(
                s_feature=dict(from_student=True, recorder='bb_s2'),
                t_feature=dict(from_student=False, recorder='bb_s2', connector='at_loss_tfeat')),
            at_loss_s3=dict(
                s_feature=dict(from_student=True, recorder='bb_s3'),
                t_feature=dict(from_student=False, recorder='bb_s3', connector='at_loss_tfeat')),
            at_loss_s4=dict(
                s_feature=dict(from_student=True, recorder='bb_s4'),
                t_feature=dict(from_student=False, recorder='bb_s4', connector='at_loss_tfeat')),
            )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
