# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class SimLoss(nn.Module):
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
        
        s_feature = torch.randn(2,16)
        t_feature = torch.randn(2,16)
        
        loss = self.loss_mse(s_feature, t_feature)

        return self.loss_weight * loss

    
