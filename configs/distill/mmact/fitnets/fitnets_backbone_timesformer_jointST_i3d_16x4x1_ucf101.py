_base_ = ['mmaction::/']

student = _base_.model
teacher_ckpt = '/'

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
        cfg_path='mmaction::/', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1.2.conv3.conv'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.3.conv3.conv'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.5.conv3.conv'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.conv3.conv')),
        teacher_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.transformer_layers.layers.1.ffns'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.transformer_layers.layers.3.ffns'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.transformer_layers.layers.9.ffns'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.transformer_layers.layers.11.ffns')),
        distill_losses=dict(
            loss_s1=dict(type='MSELoss', loss_weight=1),
            loss_s2=dict(type='MSELoss', loss_weight=1),
            loss_s3=dict(type='MSELoss', loss_weight=1),
            loss_s4=dict(type='MSELoss', loss_weight=1)),
        connectors=dict(
            loss_s1_sfeat=dict(type='CNNFeatTransfer', in_channel=256, st_size=(4,56,56), seq_len=3137, t_dim=16, embed_dim=768),
            loss_s2_sfeat=dict(type='CNNFeatTransfer', in_channel=512, st_size=(2,28,28), seq_len=3137, t_dim=16, embed_dim=768),
            loss_s3_sfeat=dict(type='CNNFeatTransfer', in_channel=1024, st_size=(2,14,14), seq_len=3137, t_dim=16, embed_dim=768),
            loss_s4_sfeat=dict(type='CNNFeatTransfer', in_channel=2048, st_size=(2,7,7), seq_len=3137, t_dim=16, embed_dim=768),
            loss_s1_tfeat=dict(type='TransFeatTransfer', in_channel=768, seq_len=3137, t_dim=16, st_size=(4,56,56), embed_dim=256, is_student=False),
            loss_s2_tfeat=dict(type='TransFeatTransfer', in_channel=768, seq_len=3137, t_dim=16, st_size=(2,28,28), embed_dim=512, is_student=False),
            loss_s3_tfeat=dict(type='TransFeatTransfer', in_channel=768, seq_len=3137, t_dim=16, st_size=(2,14,14), embed_dim=1024, is_student=False),
            loss_s4_tfeat=dict(type='TransFeatTransfer', in_channel=768, seq_len=3137, t_dim=16, st_size=(2,7,7), embed_dim=2048, is_student=False)),
        loss_forward_mappings=dict(
            loss_s1=dict(
                s_feature=dict(from_student=True, recorder='bb_s1', connector='loss_s1_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s1', connector='loss_s1_tfeat')),
            loss_s2=dict(
                s_feature=dict(from_student=True, recorder='bb_s2', connector='loss_s2_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s2', connector='loss_s2_tfeat')),
            loss_s3=dict(
                s_feature=dict(from_student=True, recorder='bb_s3', connector='loss_s3_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s3', connector='loss_s3_tfeat')),
            loss_s4=dict(
                s_feature=dict(from_student=True, recorder='bb_s4', connector='loss_s4_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s4', connector='loss_s4_tfeat')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')