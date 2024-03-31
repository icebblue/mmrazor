# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from einops import rearrange


@MODELS.register_module()
class MSELoss(nn.Module):
    """Calculate the two-norm loss between the two features.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        #print("debug:s_feature,t_feature "+ str(s_feature.size()) + " and" +str(t_feature.size()))
        s_feature = rearrange(s_feature, 'b d h w c -> b c d h w').contiguous()
        t_feature = rearrange(t_feature, 'b d h w c -> b c d h w').contiguous()
        #print("debug:s_feature,t_feature "+ str(s_feature.size()) + " and" +str(t_feature.size()))
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
            elif s_H < t_H:
                t_feature = F.adaptive_avg_pool3d(t_feature, (s_T, None, None))
        assert s_feature.size() == t_feature.size(), f"{s_feature.size()} != {t_feature.size()}"
        
        loss = self.loss_mse(s_feature, t_feature)

        return self.loss_weight * loss

    
