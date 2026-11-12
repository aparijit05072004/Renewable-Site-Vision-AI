"""SegFormer-style hierarchical transformer for 7-class land segmentation.

A faithful, CPU-trainable implementation of the SegFormer-B2 design cited in
the blueprint: Mix-Transformer (MiT) encoder with overlapping patch embedding
and spatial-reduction efficient self-attention across four stages, plus the
all-MLP lightweight decoder head.  Dimensions are scaled down ("MiT-Nano")
so fine-tuning completes on CPU; swapping ``nvidia/segformer-b2`` weights in
via HuggingFace is a drop-in replacement when a GPU is available
(see ``terraguard/data_acquisition/README``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch, stride, patch // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


class EfficientSelfAttention(nn.Module):
    """Multi-head attention with spatial-reduction (SegFormer SRA)."""

    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, c // self.heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            xr = x.permute(0, 2, 1).reshape(b, c, h, w)
            xr = self.sr(xr).reshape(b, c, -1).permute(0, 2, 1)
            xr = self.norm(xr)
        else:
            xr = x
        kv = self.kv(xr).reshape(b, -1, 2, self.heads, c // self.heads)
        k, v = kv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MixFFN(nn.Module):
    """FFN with 3x3 depthwise conv (positional information without PE)."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(b, -1, h, w)
        x = self.dw(x).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class MiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class SegFormerNano(nn.Module):
    """4-stage MiT encoder + all-MLP decoder (SegFormer architecture)."""

    def __init__(self, num_classes: int = 7,
                 dims: tuple[int, ...] = (24, 48, 96, 160),
                 depths: tuple[int, ...] = (1, 1, 2, 1),
                 heads: tuple[int, ...] = (1, 2, 4, 8),
                 sr: tuple[int, ...] = (8, 4, 2, 1),
                 decoder_dim: int = 96):
        super().__init__()
        self.stages = nn.ModuleList()
        in_ch = 3
        strides = (4, 2, 2, 2)
        patches = (7, 3, 3, 3)
        for i, dim in enumerate(dims):
            embed = OverlapPatchEmbed(in_ch, dim, patches[i], strides[i])
            blocks = nn.ModuleList(MiTBlock(dim, heads[i], sr[i])
                                   for _ in range(depths[i]))
            self.stages.append(nn.ModuleDict(
                {"embed": embed, "blocks": blocks, "norm": nn.LayerNorm(dim)}))
            in_ch = dim
        self.linear_c = nn.ModuleList(nn.Linear(d, decoder_dim) for d in dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1),
            nn.BatchNorm2d(decoder_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        size = x.shape[2:]
        feats = []
        for stage in self.stages:
            x_seq, h, w = stage["embed"](x)
            for blk in stage["blocks"]:
                x_seq = blk(x_seq, h, w)
            x_seq = stage["norm"](x_seq)
            x = x_seq.transpose(1, 2).reshape(b, -1, h, w)
            feats.append(x)
        target_hw = feats[0].shape[2:]
        fused = []
        for f, lin in zip(feats, self.linear_c):
            bb, c, h, w = f.shape
            f = lin(f.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(bb, -1, h, w)
            fused.append(F.interpolate(f, target_hw, mode="bilinear",
                                       align_corners=False))
        out = self.classifier(self.fuse(torch.cat(fused, dim=1)))
        return F.interpolate(out, size, mode="bilinear", align_corners=False)


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 7,
             classes: tuple[int, ...] | None = None) -> dict:
    """Per-class IoU and mIoU (optionally over a subset of classes)."""
    ious = {}
    for c in range(num_classes):
        p = pred == c
        t = target == c
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious[c] = (p & t).sum().item() / union
    sel = classes if classes is not None else tuple(ious)
    vals = [ious[c] for c in sel if c in ious]
    return {"per_class": ious, "miou": sum(vals) / max(len(vals), 1)}
