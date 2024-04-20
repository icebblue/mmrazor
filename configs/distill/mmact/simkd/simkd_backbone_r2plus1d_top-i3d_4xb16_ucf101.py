_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/inception_i3d/top-i3d_from-scratch_4xb16-16x4x1-100e_ucf101-rgb.py']

student = _base_.model
teacher_ckpt = '/mnt/cephfs/dataset/m3lab_data-z/pengying/checkpoints/mmaction2/r2plus1d_r34_4xb4-32x2x1-100e_ucf101-rgb/best_acc_top1_epoch_92.pth'

student["cls_head"] = dict(
    type='I3DHeadWithTransfer',
    num_classes=101,
    in_channels=1024,
    spatial_type='avg',
    dropout_ratio=0.5,
    init_std=0.01,
    average_clips='prob',
    pretrained=teacher_ckpt,
    freeze_fc=True,
    transfer_config=dict(
        type='cnn',
        s_channels=1024,
        t_channels=1024,
        factor=2,))

data_preprocessor=dict(
    type='mmaction.ActionDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    format_shape='NCTHW')

model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='SingleTeacherDistill',
    data_preprocessor=data_preprocessor,
    architecture=student,
    teacher=dict(
        cfg_path='mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_4xb4-32x2x1-100e_ucf101-rgb.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='cls_head.transfer'),),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.conv2.bn')),
        distill_losses=dict(
            loss_s4=dict(type='MSELoss', loss_weight=10)),
        loss_forward_mappings=dict(
            loss_s4=dict(
                s_feature=dict(from_student=True, recorder='bb_s4'),
                t_feature=dict(from_student=False, recorder='bb_s4')),
            )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
