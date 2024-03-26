_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/yangzehang/workplace/mmaction2/configs/recognition/i3d/i3d_k400-pretrained_4xb16-16x4x1-50e_ucf101-rgb.py']

student = _base_.model
teacher_ckpt = '/mnt/cephfs/dataset/zehang/checkpoints/mmaction2/mvit-small-p244_k400-pre_8xb4-16x4x1-50e_ucf101-rgb/best_acc_top1_epoch_10.pth'

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
        cfg_path='mmaction::/mnt/cephfs/home/zengrunhao/yangzehang/workplace/mmaction2/configs/recognition/mvit/mvit-small-p244_k400-pre_8xb4-16x4x1-50e_ucf101-rgb.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        connectors=dict(simple=dict(type='SimConnector')),
        distill_losses=dict(
            loss_kl=dict(loss_weight=1, tau=4, type='KLDivergence'),
            loss_s1=dict(loss_weight=1, type='SimLoss'),
            loss_s2=dict(loss_weight=1, type='SimLoss'),
            loss_s3=dict(loss_weight=1, type='SimLoss'),
            loss_s4=dict(loss_weight=1, type='SimLoss')),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')),
            loss_s1=dict(
                s_feature=dict(
                    connector='simple', from_student=True, recorder='bb_s1'),
                t_feature=dict(
                    connector='simple', from_student=False, recorder='bb_s4')),
            loss_s2=dict(
                s_feature=dict(
                    connector='simple', from_student=True, recorder='bb_s2'),
                t_feature=dict(
                    connector='simple', from_student=False, recorder='bb_s4')),
            loss_s3=dict(
                s_feature=dict(
                    connector='simple', from_student=True, recorder='bb_s3'),
                t_feature=dict(
                    connector='simple', from_student=False, recorder='bb_s4')),
            loss_s4=dict(
                s_feature=dict(
                    connector='simple', from_student=True, recorder='bb_s4'),
                t_feature=dict(
                    connector='simple', from_student=False,
                    recorder='bb_s4'))),
        student_recorders=dict(
            bb_s1=dict(
                source='backbone.layer1.2.conv3.conv', type='ModuleOutputs'),
            bb_s2=dict(
                source='backbone.layer2.3.conv3.conv', type='ModuleOutputs'),
            bb_s3=dict(
                source='backbone.layer3.5.conv3.conv', type='ModuleOutputs'),
            bb_s4=dict(
                source='backbone.layer4.2.conv3.conv', type='ModuleOutputs'),
            fc=dict(source='cls_head.fc_cls', type='ModuleOutputs')),
        teacher_recorders=dict(
            bb_s4=dict(
                source='backbone.blocks.15.mlp.fc2', type='ModuleOutputs'),
            fc=dict(source='cls_head.fc_cls', type='ModuleOutputs')),
        type='ConfigurableDistiller'))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=40, norm_type=2))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
