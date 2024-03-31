import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 remove_proj_q_fuse=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.remove_proj_q_fuse = remove_proj_q_fuse

        if not self.remove_proj_q_fuse:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.fuse = nn.Linear(dim, dim) 

    def forward(self, x):
        x_t, x_s = x
        B, N, C = x_t.shape

        if self.remove_proj_q_fuse:
            q = x_t.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            q = self.q(x_t).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x_s).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        a = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if not self.remove_proj_q_fuse:
            a = self.fuse(a)
        a = self.proj_drop(a)

        return a

@MODELS.register_module()
class ComKDLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        embed_dim: int = 768,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_weight = loss_weight
        self.cross_attn = CrossAttention(dim=embed_dim, num_heads=num_heads)

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        if s_feature.dim() == 5:
            s_feature = rearrange(s_feature, 'b c t h w -> b (t h w) c').contiguous()
        if t_feature.dim() == 5:
            t_feature = rearrange(t_feature, 'b c t h w -> b (t h w) c').contiguous()

        assert s_feature.dim() == 3
        assert t_feature.dim() == 3
        assert s_feature.shape == t_feature.shape

        attn = self.cross_attn((t_feature, s_feature))
        # t_feature = t_feature + attn

        # loss = self.loss_mse(s_feature, t_feature)
        loss = self.loss_mse(attn, t_feature)

        return self.loss_weight * loss