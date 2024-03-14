# Copyright (c) OpenMMLab. All rights reserved.
from .datafree_distillation import (DAFLDataFreeDistillation,
                                    DataFreeDistillation)
from .fpn_teacher_distill import FpnTeacherDistill
from .overhaul_feature_distillation import OverhaulFeatureDistillation
from .self_distill import SelfDistill
from .single_teacher_distill import SingleTeacherDistill
from .het_teacher_distill import HetTeacherDistill

__all__ = [
    'SelfDistill', 'SingleTeacherDistill', 'FpnTeacherDistill',
    'DataFreeDistillation', 'DAFLDataFreeDistillation',
    'OverhaulFeatureDistillation', 'HetTeacherDistill'
]
