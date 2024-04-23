_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/inception_i3d/top-i3d_from-scratch_4xb16-16x4x1-100e_ucf101-rgb.py']

student = _base_.model
teacher_ckpt = '/mnt/cephfs/home/zengrunhao/pengying/mmaction2/work_dirs/r2plus1d_r34_4xb4-32x2x1-100e_ucf101-rgb/best_acc_top1_epoch_3.pth'

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
        cfg_path='mmaction::/mnt/cephfs/home/zengrunhao/pengying/mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_4xb16-32x2x1-100e_ucf101-rgb.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls'),
            gt_labels=dict(type='ModuleInputs', source='cls_head.loss_cls')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        distill_losses=dict(
            loss_dkd=dict(
                type='DKDLoss',
                tau=4,
                beta=0.5,
                loss_weight=1,
                reduction='batchmean')),
        loss_forward_mappings=dict(
            loss_dkd=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc'),
                gt_labels=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
