# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector
from .utils import PatchEmbed, PatchMerging, Unpatchify, LambdaModule
from einops import rearrange


# @MODELS.register_module()
# class PatchEmbed(BaseConnector):
#     def __init__(self,
#                  in_channel: int,
#                  out_channel: int,
#                  space_stride: int = 1,
#                  time_stride: int = 1,
#                  flatten: bool = True,
#                  down_sample: bool = False,
#                  with_linear_project: bool = True,
#                  norm_layer=nn.LayerNorm,
#                  act_layer=nn.GELU
#                 ) -> None:
#         super().__init__()

#         self.flatten = flatten
#         stride = (time_stride, space_stride, space_stride)
#         self.down = nn.AvgPool3d(kernel_size=stride, stride=stride) if down_sample else nn.Identity()
#         self.proj = nn.Conv3d(in_channel, out_channel, kernel_size=stride, stride=stride) if with_linear_project else nn.Identity()
#         self.norm = norm_layer(out_channel) if norm_layer else nn.Identity()
#         self.act = act_layer() if act_layer else nn.Identity()
        
#     def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
#         x = self.proj(feature) # [B, C, T, H, W]
#         x = self.down(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)
#         x = self.act(self.norm(x))

#         return x   
    
# @MODELS.register_module()
# class PatchMerging(BaseConnector):
#     def __init__(self,
#                  in_channel: int,
#                  out_channel: int,
#                  time_size: int,
#                  space_stride: int = 1,
#                  time_stride: int = 1,
#                  rm_cls_token: bool = True,
#                  merge: bool = True,
#                  down_sample: bool = False,
#                  with_linear_project: bool = True,
#                  norm_layer = nn.BatchNorm3d,
#                  act_layer = nn.ReLU
#                 ) -> None:
#         super().__init__()

#         self.time_size = time_size
#         self.rm_cls_token = rm_cls_token
#         self.merge = merge
#         stride = (time_stride, space_stride, space_stride)
#         self.down = SequentialDownSample(time_size, space_stride, time_stride) if down_sample else nn.Identity()
#         self.proj = nn.Conv3d(in_channel, out_channel, kernel_size=stride, stride=stride) if with_linear_project else nn.Identity()
#         self.norm = norm_layer(out_channel) if norm_layer else nn.Identity()
#         self.act = act_layer() if act_layer else nn.Identity()

#     def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
#         x = feature[:, 1:] if self.rm_cls_token else feature
#         x = self.down(x)
#         if self.merge:
#             B, L, D = x.shape
#             T = self.time_size
#             H = int((L // T) ** 0.5)
#             assert T * (H ** 2) == L
#             x = x.transpose(1, 2).contiguous().reshape(B, D, T, H, H)
#         x = self.proj(x)
#         x = self.act(self.norm(x))

#         return x


# class SequentialDownSample(nn.Module):
#     def __init__(self, time_size: int, space_stride: int, time_stride: int):
#         super().__init__()
#         self.time_size = time_size
#         self.space_stride = space_stride
#         self.time_stride = time_stride

#     def forward(self, x: torch.Tensor):
#         B, L, D = x.shape
#         T = self.time_size
#         H = int((L // T) ** 0.5)
#         assert T * (H ** 2) == L

#         output_h = H // self.space_stride
#         assert output_h * self.space_stride == H
#         x = x.view(B, T, output_h, self.space_stride, output_h, self.space_stride, D)
#         x = x.mean(3).mean(4)
#         x = x.reshape(shape=(B, T, output_h ** 2, D))

#         output_t = T // self.time_stride
#         assert output_t * self.time_stride == T
#         x = x.view(B, output_t, self.time_stride, output_h ** 2, D)
#         x = x.mean(2)
#         x = x.reshape(shape=(B, output_t * output_h ** 2, D))

#         return x

# @MODELS.register_module()
# class SwinFeatureProjector(BaseConnector):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
#         return rearrange(feature, 'b d h w c -> b c d h w').contiguous()
    

@MODELS.register_module()
class CNNFeatTransfer(BaseConnector):
    def __init__(self,
                 in_channel: int = 1,
                 st_size: tuple = (1, 1, 1),
                 seq_len: int = 1,
                 t_dim: int = 1,
                 embed_dim = 1,
                 prefix_tokens: int = 1,
                 is_student: bool = True,
                ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.st_size = st_size
        self.is_student = is_student

        if is_student:
            T, H, _ = self.st_size
            grid_size = int(((seq_len - prefix_tokens) // t_dim) ** .5)

            if T >= t_dim:  # do nothing for time dimension
                if H >= grid_size:
                    patch_size = H // grid_size
                    assert patch_size * grid_size == H
                    self.aligner = PatchEmbed(H, patch_size, in_channel, embed_dim, flatten=False)
                else:
                    scale_h = grid_size // H
                    assert scale_h * H == grid_size
                    self.aligner = nn.Conv2d(in_channel, embed_dim * scale_h ** 2, 1, 1, 0)
            else:
                scale_t = t_dim // T
                assert scale_t * T == t_dim
                if H >= grid_size:
                    patch_size = H // grid_size
                    assert patch_size * grid_size == H
                    self.aligner = PatchEmbed(H, patch_size, in_channel, embed_dim * scale_t, flatten=False)
                else:
                    scale_h = grid_size // H
                    assert scale_h * H == grid_size
                    self.aligner = nn.Conv2d(in_channel, embed_dim * scale_t * scale_h ** 2, 1, 1, 0)
        else:
            self.aligner = nn.Identity()
    
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        assert self.in_channel == x.shape[1]
        assert self.st_size == tuple(x.shape[2:])
        if self.is_student:
            batches = x.shape[0]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = rearrange(self.aligner(x), '(b t) c h w -> b (t h w) c', b=batches)
        else:
            x = self.aligner(x)
        return x


@MODELS.register_module()
class TransFeatTransfer(BaseConnector):
    def __init__(self,
                 in_channel: int = 1,
                 seq_len: int = 1,
                 t_dim: int = 1,
                 st_size: tuple = (1, 1, 1),
                 embed_dim = 1,
                 prefix_tokens: int = 1,
                 is_student: bool = True,
                ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.seq_len = seq_len
        self.prefix_tokens = prefix_tokens

        T, H, _ = st_size
        grid_size = int(((seq_len - prefix_tokens) // t_dim) ** .5)

        if is_student:
            if T >= t_dim: # do nothing for time dimension
                if H >= grid_size:
                    scale = H // grid_size
                    assert scale * grid_size == H
                    self.aligner = nn.Sequential(
                        nn.Linear(in_channel, embed_dim * scale ** 2),
                        Unpatchify(t_dim, scale)
                    )
                else:
                    assert grid_size % H == 0
                    scale = grid_size // H
                    self.aligner = nn.Sequential(
                        PatchMerging(t_dim, t_dim * H ** 2, t_dim),
                        LambdaModule(lambda x: rearrange(x.view(x.shape[0], t_dim, H, H, -1), 'b t h w c -> b c t h w')),
                        nn.Conv3d(in_channel * scale ** 2, embed_dim, 1, 1, 0)
                    )      
            else:
                scale_t = t_dim // T
                assert scale_t * T == t_dim
                if H >= grid_size:
                    scale_h = H // grid_size
                    assert scale_h * grid_size == H
                    self.aligner = nn.Sequential(
                        PatchMerging(t_dim, T * grid_size ** 2, T),
                        nn.Linear(in_channel * scale_t, embed_dim * scale_h ** 2),
                        Unpatchify(T, scale_h)
                    )
                else:
                    assert grid_size % H == 0
                    scale_h = grid_size // H
                    self.aligner = nn.Sequential(
                        PatchMerging(t_dim, T * H ** 2, T),
                        LambdaModule(lambda x: rearrange(x.view(x.shape[0], T, H, H, -1), 'b t h w c -> b c t h w')),
                        nn.Conv3d(in_channel * scale_t * scale_h ** 2, embed_dim, 1, 1, 0)
                    )                       
        else:
            if T >= t_dim: # do nothing for time dimension
                if H >= grid_size:
                    self.aligner = nn.Identity()
                else:
                    self.aligner = PatchMerging(t_dim, t_dim * H ** 2, t_dim)
            else:
                if H >= grid_size:
                    self.aligner = PatchMerging(t_dim, T * grid_size ** 2, T)
                else:
                    self.aligner = PatchMerging(t_dim, T * H ** 2, T)



    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        assert self.in_channel == x.shape[-1]
        assert self.seq_len == x.shape[1]

        if self.prefix_tokens > 0:  # remove prefix tokens
            x = x[:, self.prefix_tokens:, :]

        return self.aligner(x)