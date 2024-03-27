_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/inception_i3d/top-i3d_from-scratch_4xb4-64x1x1-100e_ucf101-rgb.py']

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
            bb_s1=dict(type='ModuleOutputs', source='backbone.Conv3d_2c_3x3'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.Mixed_3c'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.Mixed_4f'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.Mixed_5c'),
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        teacher_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.Conv3d_2c_3x3'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.Mixed_3c'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.Mixed_4f'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.Mixed_5c'),
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        distill_losses=dict(
            at_loss_s1=dict(type='ATLoss', loss_weight=100),
            at_loss_s2=dict(type='ATLoss', loss_weight=100),
            at_loss_s3=dict(type='ATLoss', loss_weight=100),
            at_loss_s4=dict(type='ATLoss', loss_weight=100),
            loss_kl=dict(
                type='KLDivergence', tau=4, loss_weight=3)),
        loss_forward_mappings=dict(
            at_loss_s1=dict(
                s_feature=dict(from_student=True, recorder='bb_s1'),
                t_feature=dict(from_student=False, recorder='bb_s1')),
            at_loss_s2=dict(
                s_feature=dict(from_student=True, recorder='bb_s2'),
                t_feature=dict(from_student=False, recorder='bb_s2')),
            at_loss_s3=dict(
                s_feature=dict(from_student=True, recorder='bb_s3'),
                t_feature=dict(from_student=False, recorder='bb_s3')),
            at_loss_s4=dict(
                s_feature=dict(from_student=True, recorder='bb_s4'),
                t_feature=dict(from_student=False, recorder='bb_s4')),
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc'))
            )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
