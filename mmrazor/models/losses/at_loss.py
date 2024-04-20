# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from einops import rearrange


def orthogonal_projector(in_channels, out_channels):
    return torch.nn.utils.parametrizations.orthogonal(nn.Linear(in_channels, out_channels, bias=False).cuda())

@MODELS.register_module()
class ATLoss(nn.Module):
    """"Paying More Attention to Attention: Improving the Performance of
    Convolutional Neural Networks via Attention Transfer" Conference paper at
    ICLR2017 https://openreview.net/forum?id=Sks9_ajex.

    https://github.com/szagoruyko/attention-transfer/blob/master/utils.py

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, s_feature: torch.Tensor,
                t_feature: torch.Tensor) -> torch.Tensor:
        # print("s_feature,t_feature "+ str(s_feature.size()) + " and" +str(t_feature.size()))
        """"Forward function for ATLoss."""
        if s_feature.dim() == 4: # B x C x H x W
            s_H, t_H = s_feature.size(2), t_feature.size(2)

            if s_H > t_H:
                s_feature = F.adaptive_avg_pool2d(s_feature, (t_H, t_H))
            elif s_H < t_H:
                t_feature = F.adaptive_avg_pool2d(t_feature, (s_H, s_H))
        elif s_feature.dim() == 5: # B x C x T x H x W
            s_T, t_T = s_feature.size(2), t_feature.size(2)
            s_H, t_H = s_feature.size(3), t_feature.size(3)

            if s_H > t_H:
                s_feature = F.adaptive_avg_pool3d(s_feature, (None, t_H, t_H))
            elif s_H < t_H:
                t_feature = F.adaptive_avg_pool3d(t_feature, (None, s_H, s_H))

            if s_T > t_T:
                s_feature = F.adaptive_avg_pool3d(s_feature, (t_T, None, None))
            elif s_T < t_T:
                t_feature = F.adaptive_avg_pool3d(t_feature, (s_T, None, None))
        # print("s_feature,t_feature "+ str(s_feature.size()) + " and" +str(t_feature.size()))
        if s_feature.dim() == 4:
            loss = (self.calc_attention_matrix(s_feature) -
                    self.calc_attention_matrix(t_feature)).pow(2).mean()
        elif s_feature.dim() == 5:
            loss = (self.calc_attention_matrix(s_feature) -
                    self.calc_attention_matrix(t_feature)).pow(2).sum(1).mean()
            
        return self.loss_weight * loss

    def calc_attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """"Calculate the attention matrix.

        Args:
            x (torch.Tensor): Input features.
        """
        if x.dim() == 4: # B x C x H x W
            return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
        elif x.dim() == 5: # B x C x T x H x W
            B, T = x.size(0), x.size(2)
            return F.normalize(x.pow(2).mean(1).view(B*T, -1)).view(B, T, -1)
            
