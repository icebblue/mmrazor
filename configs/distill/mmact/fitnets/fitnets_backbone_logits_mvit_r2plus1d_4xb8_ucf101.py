_base_ = ['mmaction::/mnt/cephfs/home/zengrunhao/yangzehang/workplace/mmaction2/configs/recognition/r2plus1d/r2plus1d_r34_4xb8-16x4x1-100e_ucf101-rgb.py']

student = _base_.model
teacher_ckpt = '/mnt/cephfs/dataset/zehang/checkpoints/mmaction2/mvit-small-p244_k400-pre_4xb4-16x4x1-50e_ucf101-rgb/best_acc_top1_epoch_4.pth'

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
        cfg_path='mmaction::/mnt/cephfs/home/zengrunhao/yangzehang/workplace/mmaction2/configs/recognition/mvit/mvit-small-p244_k400-pre_4xb4-16x4x1-50e_ucf101-rgb.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.conv2.bn'),
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.blocks.15.mlp.fc2'),
            fc=dict(type='ModuleOutputs', source='cls_head.fc_cls')),
        distill_losses=dict(
            loss_s4=dict(type='MSELoss', loss_weight=3),
            loss_kl=dict(
                type='KLDivergence', tau=1, loss_weight=3)),
        connectors=dict(
            loss_s4_sfeat=dict(
                type='PatchEmbed',
                in_channel=512,
                out_channel=768,
                flatten=True),
            loss_s4_tfeat=dict(
                type='PatchMerging',
                in_channel=768,
                out_channel=768,
                time_size=8,
                time_stride=4,
                down_sample=True,
                norm_layer=None,
                act_layer=None)),
        loss_forward_mappings=dict(
            loss_s4=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s4',
                    connector='loss_s4_sfeat'),
                t_feature=dict(
                    from_student=False, 
                    recorder='bb_s4', 
                    connector='loss_s4_tfeat')),
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
