from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from mmengine.runner import load_checkpoint

from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS
from ...base import LossResults

from mmrazor.registry import MODELS
from .single_teacher_distill import SingleTeacherDistill

@MODELS.register_module()
class HetTeacherDistill(SingleTeacherDistill):
    def __init__(self,
                 distiller: dict,
                 teacher: Union[BaseModel, Dict],
                 teacher_ckpt: Optional[str] = None,
                 teacher_trainable: bool = False,
                 teacher_norm_eval: bool = True,
                 student_ckpt: Optional[str] = None,
                 student_trainable: bool = True,
                 calculate_student_loss: bool = True,
                 frames_downsample_rate: int = 1,
                 is_teacher_downsample: bool = True,
                 **kwargs) -> None:
        super().__init__(distiller, teacher, teacher_ckpt, teacher_trainable,
                         teacher_norm_eval, student_ckpt, student_trainable,
                         calculate_student_loss, **kwargs)
        
        self.frames_downsample_rate = frames_downsample_rate
        self.is_teacher_downsample = is_teacher_downsample
        
    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""

        if self.is_teacher_downsample:
            stu_batch_inputs = batch_inputs
            tea_batch_inputs = batch_inputs[..., ::self.frames_downsample_rate, :, :]
        else:
            stu_batch_inputs = batch_inputs[..., ::self.frames_downsample_rate, :, :]
            tea_batch_inputs = batch_inputs

        losses = dict()

        # If the `override_data` of a delivery is False, the delivery will
        # record the origin data.
        self.distiller.set_deliveries_override(False)
        if self.teacher_trainable:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                teacher_losses = self.teacher(
                    tea_batch_inputs, data_samples, mode='loss')

            losses.update(add_prefix(teacher_losses, 'teacher'))
        else:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                with torch.no_grad():
                    _ = self.teacher(tea_batch_inputs, data_samples, mode='loss')

        # If the `override_data` of a delivery is True, the delivery will
        # override the origin data with the recorded data.
        self.distiller.set_deliveries_override(True)
        # Original task loss will not be used during some pretraining process.
        if self.calculate_student_loss:
            with self.distiller.student_recorders, self.distiller.deliveries:
                student_losses = self.student(
                    stu_batch_inputs, data_samples, mode='loss')
            losses.update(add_prefix(student_losses, 'student'))
        else:
            with self.distiller.student_recorders, self.distiller.deliveries:
                if self.student_trainable:
                    _ = self.student(stu_batch_inputs, data_samples, mode='loss')
                else:
                    with torch.no_grad():
                        _ = self.student(
                            stu_batch_inputs, data_samples, mode='loss')

        if not self.distillation_stopped:
            # Automatically compute distill losses based on
            # `loss_forward_mappings`.
            # The required data already exists in the recorders.
            distill_losses = self.distiller.compute_distill_losses()
            losses.update(add_prefix(distill_losses, 'distill'))

        return losses