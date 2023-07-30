import math
import os
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from collections import OrderedDict

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class Conv2TranConnector(BaseConnector):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=-1,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm),
                 ) -> None:
        super().__init__()
        self.stride = stride
        self.conv_proj = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        if self.stride < 0:
            self.down_sample = nn.AvgPool2d(kernel_size=-stride, stride=-stride)
        self.ln = norm_layer(out_channel)
        self.act = act_layer()

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(feature)  # [N, C, H, W]
        if self.stride < 0:
            x = self.down_sample(x)
        else:
            H, W = x.size(2), x.size(3)
            x = F.interpolate(x, size=(H * self.stride, W * self.stride))
        x = self.act(self.ln(x.flatten(2).transpose(1, 2)))

        return x

@MODELS.register_module()
class Tran2ConvConnector(BaseConnector):
    def __init__(self,
                 in_channel,
                 out_channel,
                 hw,
                 stride=1,
                 act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d),
                 ) -> None:
        super().__init__()
        self.H, self.W = hw
        self.stride = stride
        self.conv_project = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        if self.stride < 0:
            self.down_sample = nn.AvgPool2d(kernel_size=-stride, stride=-stride)
        self.bn = norm_layer(out_channel)
        self.act = act_layer()

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        B, _, C = feature.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x = feature[:, 1:].transpose(1, 2).contiguous().reshape(B, C, self.H, self.W)
        x = self.act(self.bn(self.conv_project(x)))

        if self.stride < 0:
            x = self.down_sample(x)
        else:
            x = F.interpolate(x, size=(self.H * self.stride, self.W * self.stride))
        return x

@MODELS.register_module()
class RMClsTokenConnector(BaseConnector):
    def __init__(self,) -> None:
        super().__init__()

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:

        return feature[:, 1:]

  