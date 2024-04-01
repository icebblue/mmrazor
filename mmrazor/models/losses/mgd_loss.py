# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from einops import rearrange

@MODELS.register_module()
class MGDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation.

    <https://arxiv.org/abs/2205.01529>`

    Args:
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
    """

    def __init__(self, alpha_mgd: float = 0.00002) -> None:
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.loss_mse = nn.MSELoss(reduction='sum')

    def forward(self, preds_S: torch.Tensor,
                preds_T: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_S = rearrange(preds_S, 'b d h w c -> b c d h w').contiguous()
        preds_T = rearrange(preds_T, 'b d h w c -> b c d h w').contiguous()
# s_feature,t_feature torch.Size([4, 768, 8, 7, 7]) torch.Size([4, 768, 2, 7, 7])
# s_feature,t_feature torch.Size([4, 768, 2, 7, 7]) torch.Size([4, 768, 2, 7, 7])
        if preds_S.dim() == 4: # B x C x H x W
            s_H, t_H = preds_S.size(2), preds_T.size(2)

            if s_H > t_H:
                preds_S = F.adaptive_avg_pool2d(preds_S, (t_H, t_H))
            elif s_H < t_H:
                preds_T = F.adaptive_avg_pool2d(preds_T, (s_H, s_H))
        elif preds_S.dim() == 5: # B x C x T x H x W
            s_T, t_T = preds_S.size(2), preds_T.size(2)
            s_H, t_H = preds_S.size(3), preds_T.size(3)

            if s_H > t_H:
                preds_S = F.adaptive_avg_pool3d(preds_S, (None, t_H, t_H))
            elif s_H < t_H:
                preds_T = F.adaptive_avg_pool3d(preds_T, (None, s_H, s_H))

            if s_T > t_T:
                preds_S = F.adaptive_avg_pool3d(preds_S, (t_T, None, None))
            elif s_H < t_H:
                preds_T = F.adaptive_avg_pool3d(preds_T, (s_T, None, None))
        assert preds_S.shape == preds_T.shape,f"{preds_S.shape} != {preds_T.shape}"
        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S: torch.Tensor,
                     preds_T: torch.Tensor) -> torch.Tensor:
        """Get MSE distance of preds_S and preds_T.

        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated mse distance value.
        """
        N = preds_T.shape[0]
        dis_loss = self.loss_mse(preds_S, preds_T) / N

        return dis_loss
