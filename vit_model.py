"""
Small Vision Transformer for CIFAR-sized images, with regularization that can be changed during training.

Two regularizers, both schedulable through model.set_regularization(dropout, drop_path):
  - dropout    : one probability used at every dropout site (embedding, attention weights, attention output,
                 both MLP sites)
  - drop_path  : stochastic depth. Each residual branch (attention, MLP) is dropped for a whole sample with
                 a probability that grows linearly with depth: 0 for the first block, `drop_path` for the last.

Presets (arch names): vit_micro (~0.8M params), vit_tiny (~5.4M, DeiT-Tiny shape), vit_small (~21M).
Patch size is 4 for images up to 64 px (64 tokens at 32 px) and 16 otherwise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDropout(nn.Module):
    """Dropout whose probability can be changed during training (assign .p)."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, self.training) if self.p > 0 else x


class DropPath(nn.Module):
    """Stochastic depth: drops the whole residual branch per sample with probability p (assign .p)."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = SDropout()
        self.proj_drop = SDropout()

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.dp1 = DropPath()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), SDropout(),
                                 nn.Linear(hidden, dim), SDropout())
        self.dp2 = DropPath()

    def forward(self, x):
        x = x + self.dp1(self.attn(self.norm1(x)))
        return x + self.dp2(self.mlp(self.norm2(x)))


class ViT(nn.Module):
    def __init__(self, in_ch, img, patch, num_classes, dim, depth, heads, mlp_ratio=4.0):
        super().__init__()
        assert img % patch == 0, 'image size must be divisible by the patch size'
        n = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = SDropout()
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def set_regularization(self, dropout, drop_path):
        """Change the regularization strength during training."""
        for m in self.modules():
            if isinstance(m, SDropout):
                m.p = dropout
        depth = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            rate = drop_path * i / max(depth - 1, 1)
            blk.dp1.p = rate
            blk.dp2.p = rate

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


_PRESETS = {
    'vit_micro': dict(dim=128, depth=6, heads=4, mlp_ratio=2.0),
    'vit_tiny': dict(dim=192, depth=12, heads=3, mlp_ratio=4.0),
    'vit_small': dict(dim=384, depth=12, heads=6, mlp_ratio=4.0),
}


def get_vit(arch, in_ch, imsize, num_classes, half=False):
    patch = 4 if imsize <= 64 else 16
    model = ViT(in_ch, imsize, patch, num_classes, **_PRESETS[arch]).cuda()
    return model.half() if half else model
