# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector
from einops import rearrange


@MODELS.register_module()
class MGDConnector(BaseConnector):
    """PyTorch version of `Masked Generative Distillation.

    <https://arxiv.org/abs/2205.01529>`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
        init_cfg (Optional[Dict], optional): The weight initialized config for
            :class:`BaseModule`. Defaults to None.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        lambda_mgd: float = 0.65,
        mask_on_channel: bool = False,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.lambda_mgd = lambda_mgd
        self.mask_on_channel = mask_on_channel
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(
                teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        if self.align is not None:
            feature = self.align(feature)

        N, C, H, W = feature.shape

        device = feature.device
        if not self.mask_on_channel:
            mat = torch.rand((N, 1, H, W)).to(device)
        else:
            mat = torch.rand((N, C, 1, 1)).to(device)

        mat = torch.where(mat > 1 - self.lambda_mgd,
                          torch.zeros(1).to(device),
                          torch.ones(1).to(device)).to(device)

        masked_fea = torch.mul(feature, mat)
        new_fea = self.generation(masked_fea)
        return new_fea


@MODELS.register_module()
class MGD3DConnector(BaseConnector):
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        lambda_mgd: float = 0.65,
        mask_on_channel: bool = False,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.lambda_mgd = lambda_mgd
        self.mask_on_channel = mask_on_channel
        if student_channels != teacher_channels:
            self.align = nn.Conv3d(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv3d(
                teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                teacher_channels, teacher_channels, kernel_size=3, padding=1))
        
    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        if self.align is not None:
            feature = self.align(feature)

        N, C, T, H, W = feature.shape

        device = feature.device
        if not self.mask_on_channel:
            mat = torch.rand((N, 1, T, H, W)).to(device)
        else:
            mat = torch.rand((N, C, 1, 1, 1)).to(device)

        mat = torch.where(mat > 1 - self.lambda_mgd,
                          torch.zeros(1).to(device),
                          torch.ones(1).to(device)).to(device)
        
        masked_fea = torch.mul(feature, mat)
        new_fea = self.generation(masked_fea)
        return new_fea  


class SwinFeatureProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return rearrange(feature, 'b c d h w -> b d h w c').contiguous()
    

@MODELS.register_module()
class MGDSwinConnector(BaseConnector):
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        lambda_mgd: float = 0.65,
        mask_on_channel: bool = False,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)

        def orthogonal_projector(in_channels, out_channels):
            return torch.nn.utils.parametrizations.orthogonal(nn.Linear(in_channels, out_channels, bias=False))


        self.lambda_mgd = lambda_mgd
        self.mask_on_channel = mask_on_channel
        if student_channels != teacher_channels:
            self.align = orthogonal_projector(student_channels, teacher_channels)
        else:
            self.align = None

        self.generation = nn.Sequential(
            conv3x3(teacher_channels, teacher_channels),
            nn.ReLU(inplace=True),
            conv3x3(teacher_channels, teacher_channels))
        
        self.transfer = nn.Sequential(
            conv1x1(teacher_channels, teacher_channels),
            SwinFeatureProjector(),
            nn.LayerNorm(teacher_channels),
            nn.GELU())
        
    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        if self.align is not None:
            feature = rearrange(feature, 'b c d h w -> b d h w c').contiguous()
            feature = self.align(feature)
            feature = rearrange(feature, 'b d h w c -> b c d h w').contiguous()

        N, C, T, H, W = feature.shape

        device = feature.device
        if not self.mask_on_channel:
            mat = torch.rand((N, 1, T, H, W)).to(device)
        else:
            mat = torch.rand((N, C, 1, 1, 1)).to(device)

        mat = torch.where(mat > 1 - self.lambda_mgd,
                          torch.zeros(1).to(device),
                          torch.ones(1).to(device)).to(device)
        
        masked_fea = torch.mul(feature, mat)
        new_fea = self.transfer(self.generation(masked_fea))
        return new_fea  

@MODELS.register_module()
class OrthogonalProjectorConnector(BaseConnector):
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)

        def orthogonal_projector(in_channels, out_channels):
            return torch.nn.utils.parametrizations.orthogonal(nn.Linear(in_channels, out_channels, bias=False))

        self.align = orthogonal_projector(768, 768)


    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.align(feature)
        return feature  


@MODELS.register_module()
class atkdorthogonal(BaseConnector):
    def __init__(self) -> None:
        super().__init__()

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        schannel = feature.size(1)
        self.align = torch.nn.utils.parametrizations.orthogonal(nn.Linear(schannel, schannel, bias=False).cuda())
        feature = rearrange(feature, 'b c d h w -> b d h w c').contiguous()
        feature = self.align(feature)
        feature = rearrange(feature, 'b c d h w -> b d h w c').contiguous()
        
        return feature  
