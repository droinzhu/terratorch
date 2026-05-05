"""
Prithvi + DOFA Dual-Encoder Fusion Model Factory  v2
=====================================================
改进点 (v2):
  1. 双向跨模态注意力融合 (optical↔SAR 互相 attend，自适应门控)
  2. 光学 + SAR 联合 skip connection (中间层特征拼接后投影)
  3. 工厂新增 sar_out_indices / use_dofa_skips 参数

Architecture:
  Sentinel-2  →  Prithvi-EO-2.0-300M  ──→  intermediate tokens × 3
                                                  ↓ concat with DOFA skips
  Sentinel-1  →  DOFA-Large           ──→  intermediate tokens × 3
                                                  ↓
                                     BiCrossModalAttentionFusion  (final tokens)
                                                  ↓
                                     UNet Decoder (joint skip connections)
                                                  ↓
                                     Segmentation Head
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from terratorch.models.model import Model, ModelFactory, ModelOutput
from terratorch.models.peft_utils import get_peft_backbone
from terratorch.registry import MODEL_FACTORY_REGISTRY, TERRATORCH_BACKBONE_REGISTRY


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _remove_cls(x: torch.Tensor) -> torch.Tensor:
    """[B, N+1, D] → [B, N, D]  (drop CLS token at index 0)"""
    return x[:, 1:, :]


def _tokens_to_spatial(x: torch.Tensor) -> torch.Tensor:
    """[B, N, D] → [B, D, H, W]  (assumes H == W == sqrt(N))"""
    B, N, D = x.shape
    H = W = int(math.sqrt(N))
    assert H * W == N, f"Token count {N} is not a perfect square."
    return x.permute(0, 2, 1).reshape(B, D, H, W)


# ---------------------------------------------------------------------------
# 双向跨模态注意力融合 (v2: 自适应门控)
# ---------------------------------------------------------------------------

class CrossModalAttentionFusion(nn.Module):
    """
    双向跨模态融合：
      - 光学 tokens 作 Query，SAR 作 KV  (o2s)
      - SAR tokens 作 Query，光学 作 KV  (s2o)
      - 门控网络自适应加权两个方向的输出
      - FFN + 残差

    Args:
        dim:       Token 维度 (default: 1024)
        num_heads: 注意力头数 (default: 8)
        dropout:   Dropout 概率 (default: 0.1)
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_opt = nn.LayerNorm(dim)
        self.norm_sar = nn.LayerNorm(dim)
        # 方向1: 光学 Q → SAR KV
        self.cross_attn_o2s = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # 方向2: SAR Q → 光学 KV
        self.cross_attn_s2o = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # 自适应门控：根据两个方向的输出决定权重
        self.gate = nn.Linear(dim * 2, dim)
        # FFN
        self.norm_post = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            optical: [B, N, dim]
            sar:     [B, N, dim]
        Returns:
            fused:   [B, N, dim]
        """
        opt_n = self.norm_opt(optical)
        sar_n = self.norm_sar(sar)

        # 方向1: 光学关注SAR细节
        o2s, _ = self.cross_attn_o2s(query=opt_n, key=sar_n, value=sar)
        # 方向2: SAR关注光学语义 (SAR对洪水更敏感，这方向很重要)
        s2o, _ = self.cross_attn_s2o(query=sar_n, key=opt_n, value=optical)

        # 自适应门控融合
        gate = torch.sigmoid(self.gate(torch.cat([o2s, s2o], dim=-1)))  # [B,N,dim]
        fused = gate * o2s + (1.0 - gate) * s2o

        # 残差 + FFN
        x = optical + fused
        x = x + self.ffn(self.norm_post(x))
        return x


# ---------------------------------------------------------------------------
# Skip Projection  (tokens → spatial feature map)
# ---------------------------------------------------------------------------

class SkipProjection(nn.Module):
    """
    将 token 特征投影到空间特征图，用于 UNet 跳跃连接。
    支持单模态 (in_dim) 或拼接双模态 (in_dim*2) 输入。
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, in_dim] → [B, out_dim, H, W]"""
        return _tokens_to_spatial(self.proj(tokens))


# ---------------------------------------------------------------------------
# UNet Decoder Block
# ---------------------------------------------------------------------------

class UpsampleBlock(nn.Module):
    """Upsample ×2 → (optional cat with skip) → Conv-BN-ReLU ×2."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int = 0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full Fusion Model
# ---------------------------------------------------------------------------

class PrithviDofaFusionModel(Model):
    """
    双编码器分割模型 (v2)。

    Forward 输入:
        x (dict):  {"S2": Tensor[B, C_s2, H, W], "S1": Tensor[B, C_s1, H, W]}

    Forward 输出:
        ModelOutput with output Tensor[B, num_classes, H, W]
    """

    def __init__(
        self,
        optical_encoder: nn.Module,
        sar_encoder: nn.Module,
        fusion: CrossModalAttentionFusion,
        skip_projs: nn.ModuleList,
        decoder_blocks: nn.ModuleList,
        seg_head: nn.Module,
        image_size: int,
        s2_modality_key: str,
        s1_modality_key: str,
        use_dofa_skips: bool = True,
    ):
        super().__init__()
        self.optical_encoder = optical_encoder
        self.sar_encoder = sar_encoder
        self.fusion = fusion
        self.skip_projs = skip_projs
        self.decoder_blocks = decoder_blocks
        self.seg_head = seg_head
        self.image_size = image_size
        self.s2_key = s2_modality_key
        self.s1_key = s1_modality_key
        self.use_dofa_skips = use_dofa_skips

    def freeze_encoder(self):
        for p in self.optical_encoder.parameters():
            p.requires_grad_(False)
        for p in self.sar_encoder.parameters():
            p.requires_grad_(False)

    def freeze_decoder(self):
        for mod in [self.fusion, self.skip_projs, self.decoder_blocks, self.seg_head]:
            for p in mod.parameters():
                p.requires_grad_(False)

    def forward(self, x: Union[Dict, torch.Tensor], **kwargs) -> ModelOutput:
        if not isinstance(x, dict):
            raise TypeError(
                "PrithviDofaFusionModel 期望字典输入: "
                f"{{'{self.s2_key}': Tensor, '{self.s1_key}': Tensor}}. 实际: {type(x)}."
            )

        s2 = x[self.s2_key]   # [B, C_s2, H, W]
        s1 = x[self.s1_key]   # [B, C_s1, H, W]

        # ── 光学分支 (Prithvi) ─────────────────────────────────────────
        optical_feats: list = self.optical_encoder(s2, **kwargs)
        optical_final = _remove_cls(optical_feats[-1])          # [B, N, D]
        optical_skips = [_remove_cls(f) for f in optical_feats[:-1]]

        # ── SAR 分支 (DOFA) ────────────────────────────────────────────
        sar_feats: list = self.sar_encoder(s1)
        sar_final = _remove_cls(sar_feats[-1])                  # [B, N, D]
        sar_skips = [_remove_cls(f) for f in sar_feats[:-1]] if self.use_dofa_skips else []

        # ── 双向跨模态融合 ─────────────────────────────────────────────
        fused = self.fusion(optical_final, sar_final)           # [B, N, D]
        dec = _tokens_to_spatial(fused)                         # [B, D, 14, 14]

        # ── 构建 skip connections (光学 + SAR 拼接) ────────────────────
        n_skip = len(self.skip_projs)
        # 光学中间层：取最后 n_skip 个，从深到浅
        opt_skips = list(reversed(optical_skips[-n_skip:])) if optical_skips else []
        # SAR 中间层：同样取最后 n_skip 个
        sar_skips_sel = list(reversed(sar_skips[-n_skip:])) if sar_skips else []

        # ── UNet 解码 ──────────────────────────────────────────────────
        for i, block in enumerate(self.decoder_blocks):
            if i < len(opt_skips):
                opt_tok = opt_skips[i]
                if i < len(sar_skips_sel):
                    # 拼接光学和SAR token特征
                    combined = torch.cat([opt_tok, sar_skips_sel[i]], dim=-1)
                else:
                    combined = opt_tok
                skip = self.skip_projs[i](combined)
            else:
                skip = None
            dec = block(dec, skip)

        # ── 插值到输入尺寸 ────────────────────────────────────────────
        if dec.shape[-1] != self.image_size:
            dec = F.interpolate(
                dec, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )

        # ── 分割头 ────────────────────────────────────────────────────
        mask = self.seg_head(dec)   # [B, num_classes, H, W]
        return ModelOutput(output=mask)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

@MODEL_FACTORY_REGISTRY.register
class PrithviDofaModelFactory(ModelFactory):
    """
    构建 Prithvi + DOFA 双编码器融合分割模型的工厂类 (v2)。

    Usage in YAML::

        model:
          class_path: terratorch.tasks.SemanticSegmentationTask
          init_args:
            model_factory: PrithviDofaModelFactory
            model_args:
              num_classes: 2
              ...
    """

    def build_model(
        self,
        task: str = "segmentation",
        num_classes: int = 2,
        # ── 光学编码器 (Prithvi) ─────────────────────────────────────
        optical_backbone: str = "prithvi_eo_v2_300",
        optical_backbone_kwargs: Optional[Dict] = None,
        optical_out_indices: Optional[List[int]] = None,
        optical_peft_config: Optional[Dict] = None,
        # ── SAR 编码器 (DOFA) ────────────────────────────────────────
        sar_backbone: str = "dofa_large_patch16_224",
        sar_backbone_kwargs: Optional[Dict] = None,
        sar_out_indices: Optional[List[int]] = None,
        sar_peft_config: Optional[Dict] = None,
        # ── 融合 ────────────────────────────────────────────────────
        embed_dim: int = 1024,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
        # ── UNet 解码器 ──────────────────────────────────────────────
        decoder_channels: Optional[List[int]] = None,
        skip_dim: int = 256,
        num_skip_levels: int = 3,
        use_dofa_skips: bool = True,
        # ── 其他 ────────────────────────────────────────────────────
        image_size: int = 224,
        freeze_encoders: bool = False,
        s2_modality_key: str = "S2",
        s1_modality_key: str = "S1",
        **kwargs,
    ) -> PrithviDofaFusionModel:
        """
        构建并返回 PrithviDofaFusionModel (v2)。

        新增参数:
            sar_out_indices:   DOFA 中间层索引，默认 [5,11,17,23]（与 Prithvi 对齐）。
            use_dofa_skips:    是否将 DOFA 中间层特征与 Prithvi 拼接用于 skip。
                               True 时 SkipProjection 输入维度为 embed_dim*2。
        """
        if task.lower() != "segmentation":
            raise NotImplementedError(
                f"PrithviDofaModelFactory 仅支持 'segmentation'，当前: '{task}'."
            )

        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]

        if optical_out_indices is None:
            optical_out_indices = [5, 11, 17, 23]

        if sar_out_indices is None:
            sar_out_indices = [5, 11, 17, 23]  # 与 Prithvi 对齐，获取多尺度特征

        # ── 构建光学编码器 ────────────────────────────────────────────
        optical_kwargs = dict(optical_backbone_kwargs or {})
        optical_kwargs.setdefault("out_indices", optical_out_indices)
        optical_kwargs.setdefault("encoder_only", True)

        optical_enc = TERRATORCH_BACKBONE_REGISTRY.build(optical_backbone, **optical_kwargs)

        if optical_peft_config is not None:
            if not optical_kwargs.get("pretrained", False):
                warnings.warn("对光学编码器应用 PEFT 但未加载预训练权重。", stacklevel=1)
            optical_enc = get_peft_backbone(optical_peft_config, optical_enc)

        # ── 构建 SAR 编码器 ───────────────────────────────────────────
        sar_kwargs = dict(sar_backbone_kwargs or {})
        sar_kwargs.setdefault("out_indices", sar_out_indices)

        sar_enc = TERRATORCH_BACKBONE_REGISTRY.build(sar_backbone, **sar_kwargs)

        if sar_peft_config is not None:
            if not sar_kwargs.get("pretrained", False):
                warnings.warn("对 SAR 编码器应用 PEFT 但未加载预训练权重。", stacklevel=1)
            sar_enc = get_peft_backbone(sar_peft_config, sar_enc)

        # ── 双向跨模态融合 ─────────────────────────────────────────────
        fusion = CrossModalAttentionFusion(
            dim=embed_dim, num_heads=fusion_heads, dropout=fusion_dropout
        )

        # ── Skip 投影 ─────────────────────────────────────────────────
        # use_dofa_skips=True 时，skip 输入是光学+SAR拼接，维度为 embed_dim*2
        skip_in_dim = embed_dim * 2 if use_dofa_skips else embed_dim
        skip_projs = nn.ModuleList(
            [SkipProjection(skip_in_dim, skip_dim) for _ in range(num_skip_levels)]
        )

        # ── UNet 解码器 ───────────────────────────────────────────────
        decoder_blocks = nn.ModuleList()
        in_ch = embed_dim
        for i, out_ch in enumerate(decoder_channels):
            s_ch = skip_dim if i < num_skip_levels else 0
            decoder_blocks.append(UpsampleBlock(in_ch, out_ch, skip_ch=s_ch))
            in_ch = out_ch

        # ── 分割头 ────────────────────────────────────────────────────
        seg_head = nn.Conv2d(in_ch, num_classes, kernel_size=1)

        # ── 组装模型 ──────────────────────────────────────────────────
        model = PrithviDofaFusionModel(
            optical_encoder=optical_enc,
            sar_encoder=sar_enc,
            fusion=fusion,
            skip_projs=skip_projs,
            decoder_blocks=decoder_blocks,
            seg_head=seg_head,
            image_size=image_size,
            s2_modality_key=s2_modality_key,
            s1_modality_key=s1_modality_key,
            use_dofa_skips=use_dofa_skips,
        )

        if freeze_encoders:
            model.freeze_encoder()

        return model
