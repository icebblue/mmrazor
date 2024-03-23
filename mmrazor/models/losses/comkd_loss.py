import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from einops import rearrange

@MODELS.register_module()
class ComKDLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        in_channels: int = 768,
        out_channels: int = 768,
    ) -> None:
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_weight = loss_weight
        self.proj_q = nn.Linear(in_channels, out_channels)
        self.proj_o = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        if s_feature.dim() == 5:
            s_feature = rearrange(s_feature, 'b c t h w -> b (t h w) c').contiguous()
        if t_feature.dim() == 5:
            s_feature = rearrange(t_feature, 'b c t h w -> b (t h w) c').contiguous()

        assert s_feature.dim() == 3
        assert t_feature.dim() == 3
        assert s_feature.shape == t_feature.shape

        feat_q = self.proj_q(t_feature)
        C = feat_q.shape[-1]
        attn_score = torch.matmul(feat_q, s_feature.transpose(-1, -2))  # [B, L, C] * [B, C, L] => [B, L, L]
        attn_score = attn_score / math.sqrt(C)
        attn_prob = nn.Softmax(dim=-1)(attn_score)
        attn_map = torch.matmul(attn_prob, s_feature)
        attn_map = self.proj_o(attn_map)
            
        t_feature = t_feature + attn_map

        loss = self.loss_mse(s_feature, t_feature)

        return self.loss_weight * loss