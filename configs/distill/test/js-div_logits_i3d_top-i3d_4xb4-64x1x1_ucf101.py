_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/inception_i3d/top-i3d_js-div_test.py']

student = _base_.model
teacher_ckpt = '/mnt/cephfs/dataset/zehang/checkpoints/mmaction2/i3d_imagenet-pre_4xb4-64x1x1-100e_ucf101-rgb/best_acc_top1_epoch_64.pth'
student_ckpt = '/mnt/cephfs/dataset/zehang/checkpoints/mmrazor/atkd_backbone_i3d_top-i3d_4xb4-64x1x1_ucf101/best_acc_top1_epoch_83.pth'

# student["cls_head"] = dict(
#     type='I3DHeadWithTransfer',
#     num_classes=101,
#     in_channels=1024,
#     spatial_type='avg',
#     dropout_ratio=0.5,
#     init_std=0.01,
#     average_clips='prob',
#     pretrained=teacher_ckpt,
#     freeze_fc=True,
#     transfer_config=dict(
#         type='cnn',
#         s_channels=1024,
#         t_channels=1024,
#         factor=2,))

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
    student_ckpt=student_ckpt,
    student_trainable=False,
    calculate_student_loss=False,
    teacher=dict(
        cfg_path='mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/inception_i3d/i3d_imagenet-pre_4xb4-64x1x1-100e_ucf101-rgb.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        distill_losses=dict(
            loss_kl=dict(type='JSDivergence', loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')