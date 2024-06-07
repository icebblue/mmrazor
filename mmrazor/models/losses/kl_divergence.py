# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        tau_T: float = 1.0,
        tau_S: float = 1.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence, self).__init__()
        self.tau_T = tau_T
        self.tau_S = tau_S
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if self.teacher_detach:
            preds_T = preds_T.detach()
        # max_values, _ = torch.max(preds_T, dim=1)
        # # 找出最大值大于等于0.8的索引
        # indices = torch.nonzero((max_values <= 0.9) &(max_values>=0.6 ) ).squeeze()

        # tmp = preds_S.clone()
        # 找出每行中的最大值和第二大值
        max_values, max_indices = torch.topk(preds_T, k=2, dim=1)
        # 计算最大值减去第二大值
        diff = max_values[:, 0] - max_values[:, 1]
        # 找出最大值减去第二大值在0.6到0.8之间的索引
        indices = torch.nonzero((diff > 0.6) & (diff <= 0.8)).squeeze()
        # 提取需要处理的部分
        tensor_to_process = preds_T[indices]
        # 对需要处理的部分进行操作，例如除以3.2
        tensor_to_process /= 0.7
        # 将处理后的部分重新放回原张量的相应位置
        preds_T[indices] = tensor_to_process

        # tmp = preds_S.clone()
        # if self.teacher_detach:
        #     preds_T = preds_T.detach()
        # max_values, _ = torch.max(tmp, dim=1)
        # # 找出最大值大于等于0.8的索引
        # indices = torch.nonzero(max_values <= 0.98 ).squeeze()
        # # 提取需要处理的部分
        # tensor_to_process = tmp[indices]
        # # 对需要处理的部分进行操作，例如除以3.2
        # tensor_to_process /= 1.2
        # # 将处理后的部分重新放回原张量的相应位置
        # tmp[indices] = tensor_to_process

        softmax_pred_T = F.softmax(preds_T / self.tau_T, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau_S, dim=1)
        loss = (self.tau_S**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return self.loss_weight * loss
