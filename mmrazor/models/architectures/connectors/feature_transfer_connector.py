# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector
from einops import rearrange


@MODELS.register_module()
class PatchEmbed(BaseConnector):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 space_stride: int = 1,
                 time_stride: int = 1,
                 flatten: bool = True,
                 down_sample: bool = False,
                 with_linear_project: bool = True,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU
                ) -> None:
        super().__init__()

        self.flatten = flatten
        stride = (time_stride, space_stride, space_stride)
        self.down = nn.AvgPool3d(kernel_size=stride, stride=stride) if down_sample else nn.Identity()
        self.proj = nn.Conv3d(in_channel, out_channel, kernel_size=stride, stride=stride) if with_linear_project else nn.Identity()
        self.norm = norm_layer(out_channel) if norm_layer else nn.Identity()
        self.act = act_layer() if act_layer else nn.Identity()
        
    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.proj(feature) # [B, C, T, H, W]
        x = self.down(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.act(self.norm(x))

        return x   
    
@MODELS.register_module()
class PatchMerging(BaseConnector):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 time_size: int,
                 space_stride: int = 1,
                 time_stride: int = 1,
                 rm_cls_token: bool = True,
                 merge: bool = True,
                 down_sample: bool = False,
                 with_linear_project: bool = True,
                 norm_layer = nn.BatchNorm3d,
                 act_layer = nn.ReLU
                ) -> None:
        super().__init__()

        self.time_size = time_size
        self.rm_cls_token = rm_cls_token
        self.merge = merge
        stride = (time_stride, space_stride, space_stride)
        self.down = SequentialDownSample(time_size, space_stride, time_stride) if down_sample else nn.Identity()
        self.proj = nn.Conv3d(in_channel, out_channel, kernel_size=stride, stride=stride) if with_linear_project else nn.Identity()
        self.norm = norm_layer(out_channel) if norm_layer else nn.Identity()
        self.act = act_layer() if act_layer else nn.Identity()

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        x = feature[:, 1:] if self.rm_cls_token else feature
        x = self.down(x)
        if self.merge:
            B, L, D = x.shape
            T = self.time_size
            H = int((L // T) ** 0.5)
            assert T * (H ** 2) == L
            x = x.transpose(1, 2).contiguous().reshape(B, D, T, H, H)
        x = self.proj(x)
        x = self.act(self.norm(x))

        return x


class SequentialDownSample(nn.Module):
    def __init__(self, time_size: int, space_stride: int, time_stride: int):
        super().__init__()
        self.time_size = time_size
        self.space_stride = space_stride
        self.time_stride = time_stride

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        T = self.time_size
        H = int((L // T) ** 0.5)
        assert T * (H ** 2) == L

        output_h = H // self.space_stride
        assert output_h * self.space_stride == H
        x = x.view(B, T, output_h, self.space_stride, output_h, self.space_stride, D)
        x = x.mean(3).mean(4)
        x = x.reshape(shape=(B, T, output_h ** 2, D))

        output_t = T // self.time_stride
        assert output_t * self.time_stride == T
        x = x.view(B, output_t, self.time_stride, output_h ** 2, D)
        x = x.mean(2)
        x = x.reshape(shape=(B, output_t * output_h ** 2, D))

        return x

@MODELS.register_module()
class SwinFeatureProjector(BaseConnector):
    def __init__(self) -> None:
        super().__init__()

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        return rearrange(feature, 'b d h w c -> b c d h w').contiguous()