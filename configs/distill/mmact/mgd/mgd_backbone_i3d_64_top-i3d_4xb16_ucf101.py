_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/inception_i3d/top-i3d_from-scratch_4xb16-16x4x1-100e_ucf101-rgb.py']

student = _base_.model
teacher_ckpt = '/mnt/cephfs/dataset/m3lab_data-z/pengying/checkpoints/mmaction2/i3d_imagenet-pre_4xb4-64x1x1-100e_ucf101-rgb/best_acc_top1_epoch_64.pth'

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
        cfg_path='mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/inception_i3d/i3d_imagenet-pre_4xb4-64x1x1-100e_ucf101-rgb.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.Mixed_5c')),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.Mixed_5c')),
        distill_losses=dict(loss_mgd=dict(type='MGDLoss', alpha_mgd=0.00004)),
        connectors=dict(
            loss_mgd_sfeat=dict(
                type='MGD3DConnector',
                student_channels=1024,
                teacher_channels=1024)),
        loss_forward_mappings=dict(
            loss_mgd=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='bb_s4',
                    connector='loss_mgd_sfeat'),
                preds_T=dict(
                    from_student=False,
                    recorder='bb_s4')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
