_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/yangzehang/workplace/mmaction2/configs/recognition/inception_i3d/top-i3d_from-scratch_4xb4-64x1x1-100e_ucf101-rgb.py']

student = _base_.model
teacher_ckpt = '/mnt/cephfs/dataset/zehang/checkpoints/mmaction2/swin-tiny-p244-w877_k400-pre_4xb4-16x4x1-50e_ucf101-rgb/best_acc_top1_epoch_21.pth'

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
        cfg_path='mmaction::/mnt/cephfs/home/zengrunhao/yangzehang/workplace/mmaction2/configs/recognition/swin/swin-tiny-p244-w877_k400-pre_4xb4-16x4x1-50e_ucf101-rgb.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    frames_downsample_rate=4,
    is_teacher_downsample=True,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        distill_losses=dict(
            loss_kl=dict(
                type='CTKD', 
                cosine_decay=True,
                decay_max=0,
                decay_min=-1,
                decay_loops=5,
                max_iter=597,
                loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
