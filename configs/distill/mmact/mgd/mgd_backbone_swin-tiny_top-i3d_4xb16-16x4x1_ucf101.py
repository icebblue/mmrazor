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
            bb_s4=dict(type='ModuleOutputs', source='backbone.Mixed_5c')),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layers.3.blocks.1.mlp')),
        distill_losses=dict(loss_mgd=dict(type='MGDLoss', alpha_mgd=0.000002)),
        connectors=dict(
            loss_mgd_sfeat=dict(
                type='MGDSwinConnector',
                student_channels=1024,
                teacher_channels=768)),
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
