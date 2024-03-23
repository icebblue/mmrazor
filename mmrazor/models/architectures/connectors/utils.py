import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = (img_size,) * 2
        patch_size = (patch_size,) * 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, t_dim, out_seq_len, out_t_dim):
        super().__init__()
        self.t_dim = t_dim
        self.out_seq_len = out_seq_len
        self.out_t_dim = out_t_dim

    def forward(self, x):
        B, seq_len, C = x.shape

        grid_size = int((seq_len // self.t_dim) ** 0.5)
        assert grid_size ** 2 * self.t_dim == seq_len

        out_grid_size = int((self.out_seq_len // self.out_t_dim) ** 0.5)
        assert out_grid_size ** 2 * self.out_t_dim == self.out_seq_len
            
        scale_t = self.t_dim // self.out_t_dim
        assert scale_t * self.out_t_dim == self.t_dim
        scale_h = grid_size // out_grid_size
        assert scale_h * out_grid_size == grid_size
        x = x.view(B, self.out_t_dim, scale_t, out_grid_size, scale_h, out_grid_size, scale_h, C)
        x = rearrange(x, 'b t o h p w q c -> b (t h w) (o p q c)')
        return x


class Unpatchify(nn.Module):
    def __init__(self, t_dim, scale):
        super().__init__()
        self.t_dim = t_dim
        self.scale = scale

    def forward(self, x):
        B, seq_len, _ = x.shape
        H = W = int((seq_len // self.t_dim) ** 0.5)
        assert H * W * self.t_dim == seq_len
        x = x.view(B, self.t_dim, H, W, self.scale, self.scale, -1)
        x = rearrange(x, 'b t h w p q c -> b c t (h p) (w q)')
        return x
    

class LambdaModule(nn.Module):
    def __init__(self, lambda_fn):
        super(LambdaModule, self).__init__()
        self.fn = lambda_fn

    def forward(self, x):
        return self.fn(x)