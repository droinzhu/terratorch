"""Comprehensive tests for prithvi_vit.py to maximize coverage."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_vit import (
    PRETRAINED_BANDS,
    PRITHVI_V1_MEAN,
    PRITHVI_V1_STD,
    PRITHVI_V2_MEAN,
    PRITHVI_V2_STD,
    _cfg,
    _create_prithvi,
    checkpoint_filter_fn_mae,
    checkpoint_filter_fn_vit,
    checkpoint_filter_fn_vit_adapter,
    prithvi_adapter_cfgs,
    prithvi_cfgs,
    pretrained_weights,
    prithvi_eo_tiny,
    prithvi_eo_v1_100,
    prithvi_eo_v1_100_mae,
    prithvi_eo_v2_100_tl,
    prithvi_eo_v2_100_tl_mae,
    prithvi_eo_v2_300,
    prithvi_eo_v2_300_mae,
    prithvi_eo_v2_300_tl,
    prithvi_eo_v2_300_tl_mae,
    prithvi_eo_v2_600,
    prithvi_eo_v2_600_mae,
    prithvi_eo_v2_600_tl,
    prithvi_eo_v2_600_tl_mae,
    prithvi_eo_v2_tiny_tl,
    prithvi_eo_v2_tiny_tl_mae,
)


# Test constants
def test_pretrained_bands_structure():
    """Test PRETRAINED_BANDS constant."""
    assert len(PRETRAINED_BANDS) == 6
    assert all(isinstance(band, HLSBands) for band in PRETRAINED_BANDS)
    assert HLSBands.BLUE in PRETRAINED_BANDS
    assert HLSBands.GREEN in PRETRAINED_BANDS


def test_prithvi_v1_mean_std():
    """Test Prithvi V1 mean and std constants."""
    assert len(PRITHVI_V1_MEAN) == 6
    assert len(PRITHVI_V1_STD) == 6
    assert all(isinstance(v, (int, float)) for v in PRITHVI_V1_MEAN)
    assert all(isinstance(v, (int, float)) for v in PRITHVI_V1_STD)


def test_prithvi_v2_mean_std():
    """Test Prithvi V2 mean and std constants."""
    assert len(PRITHVI_V2_MEAN) == 6
    assert len(PRITHVI_V2_STD) == 6
    assert all(isinstance(v, (int, float)) for v in PRITHVI_V2_MEAN)
    assert all(isinstance(v, (int, float)) for v in PRITHVI_V2_STD)


def test_cfg_default():
    """Test _cfg function with default parameters."""
    cfg = _cfg()
    
    assert cfg["img_size"] == 224
    assert cfg["num_frames"] == 4
    assert cfg["patch_size"] == [1, 16, 16]
    assert cfg["in_chans"] == 6
    assert cfg["embed_dim"] == 768
    assert cfg["mean"] == PRITHVI_V2_MEAN
    assert cfg["std"] == PRITHVI_V2_STD


def test_cfg_with_overrides():
    """Test _cfg function with custom parameters."""
    cfg = _cfg(embed_dim=1024, depth=24, custom_param="test")
    
    assert cfg["embed_dim"] == 1024
    assert cfg["depth"] == 24
    assert cfg["custom_param"] == "test"


def test_prithvi_cfgs_structure():
    """Test prithvi_cfgs dictionary structure."""
    assert "prithvi_eo_tiny" in prithvi_cfgs
    assert "prithvi_eo_v1_100" in prithvi_cfgs
    assert "prithvi_eo_v2_300" in prithvi_cfgs
    assert "prithvi_eo_v2_600_tl" in prithvi_cfgs
    
    # Check each config has required keys
    for variant, cfg in prithvi_cfgs.items():
        assert "embed_dim" in cfg
        assert "depth" in cfg
        assert "num_heads" in cfg


def test_pretrained_weights_structure():
    """Test pretrained_weights dictionary structure."""
    assert "prithvi_eo_v1_100" in pretrained_weights
    assert "prithvi_eo_v2_300" in pretrained_weights
    
    for variant, info in pretrained_weights.items():
        assert "hf_hub_id" in info
        assert "hf_hub_filename" in info


def test_prithvi_adapter_cfgs_structure():
    """Test prithvi_adapter_cfgs dictionary structure."""
    assert "prithvi_eo_v1_100" in prithvi_adapter_cfgs
    assert "prithvi_eo_v2_300" in prithvi_adapter_cfgs
    
    for variant, cfg in prithvi_adapter_cfgs.items():
        assert "interaction_indexes" in cfg
        assert "conv_inplane" in cfg
        assert "deform_num_heads" in cfg


# Test checkpoint filter functions
def test_checkpoint_filter_fn_vit_basic():
    """Test checkpoint_filter_fn_vit with basic state dict."""
    from terratorch.models.backbones.prithvi_mae import PrithviViT
    
    model = MagicMock(spec=PrithviViT)
    model.pos_embed = torch.randn(1, 100, 768)
    model.temporal_encoding = False
    model.location_encoding = False
    model.state_dict.return_value = {"patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16)}
    
    state_dict = {
        "patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16),
        "blocks.0.weight": torch.randn(768, 768),
        "pos_embed": torch.randn(1, 100, 768),
        "decoder.weight": torch.randn(512, 768),  # Should be dropped
        "mask_token": torch.randn(768),  # Should be dropped
    }
    
    result = checkpoint_filter_fn_vit(state_dict, model, PRETRAINED_BANDS, PRETRAINED_BANDS)
    
    assert "blocks.0.weight" in result
    assert "decoder.weight" not in result
    assert "mask_token" not in result


def test_checkpoint_filter_fn_vit_with_timm_module():
    """Test checkpoint_filter_fn_vit with _timm_module prefix."""
    from terratorch.models.backbones.prithvi_mae import PrithviViT
    
    model = MagicMock(spec=PrithviViT)
    model.pos_embed = torch.randn(1, 100, 768)
    model.temporal_encoding = False
    model.location_encoding = False
    model.state_dict.return_value = {"patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16)}
    
    state_dict = {
        "_timm_module.patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16),
        "_timm_module.blocks.0.weight": torch.randn(768, 768),
    }
    
    result = checkpoint_filter_fn_vit(state_dict, model, PRETRAINED_BANDS, PRETRAINED_BANDS)
    
    # _timm_module prefix should be removed
    assert any("blocks" in k for k in result.keys())
    assert not any("_timm_module" in k for k in result.keys())


def test_checkpoint_filter_fn_vit_with_encoder_prefix():
    """Test checkpoint_filter_fn_vit with encoder prefix."""
    from terratorch.models.backbones.prithvi_mae import PrithviViT
    
    model = MagicMock(spec=PrithviViT)
    model.pos_embed = torch.randn(1, 100, 768)
    model.temporal_encoding = False
    model.location_encoding = False
    model.state_dict.return_value = {"patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16)}
    
    state_dict = {
        "encoder.patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16),
        "encoder.blocks.0.weight": torch.randn(768, 768),
    }
    
    result = checkpoint_filter_fn_vit(state_dict, model, PRETRAINED_BANDS, PRETRAINED_BANDS)
    
    # encoder prefix should be removed for ViT
    assert any("blocks" in k for k in result.keys())
    assert not any("encoder." in k for k in result.keys())


def test_checkpoint_filter_fn_vit_with_vpt():
    """Test checkpoint_filter_fn_vit with VPT prompt embeddings."""
    from terratorch.models.backbones.prithvi_mae import PrithviViT
    
    model = MagicMock(spec=PrithviViT)
    model.pos_embed = torch.randn(1, 100, 768)
    model.temporal_encoding = False
    model.location_encoding = False
    vpt_tensor = torch.randn(10, 768)
    model.state_dict.return_value = {
        "vpt_prompt_embeddings": vpt_tensor,
        "patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16)
    }
    
    state_dict = {
        "patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16),
        "blocks.0.weight": torch.randn(768, 768),
    }
    
    result = checkpoint_filter_fn_vit(state_dict, model, PRETRAINED_BANDS, PRETRAINED_BANDS)
    
    assert "vpt_prompt_embeddings" in result
    assert torch.equal(result["vpt_prompt_embeddings"], vpt_tensor)


def test_checkpoint_filter_fn_mae_basic():
    """Test checkpoint_filter_fn_mae with basic state dict."""
    from terratorch.models.backbones.prithvi_mae import PrithviMAE
    
    model = MagicMock(spec=PrithviMAE)
    model.encoder = MagicMock()
    model.encoder.pos_embed = torch.randn(1, 100, 768)
    model.encoder.temporal_encoding = False
    model.encoder.location_encoding = False
    model.decoder = MagicMock()
    model.decoder.decoder_pos_embed = torch.randn(1, 50, 512)
    model.state_dict.return_value = {"encoder.patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16)}
    
    state_dict = {
        "encoder.patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16),
        "decoder.blocks.0.weight": torch.randn(512, 512),
        "pos_embed": torch.randn(1, 100, 768),
        "decoder_pos_embed": torch.randn(1, 50, 512),
    }
    
    result = checkpoint_filter_fn_mae(state_dict, model, PRETRAINED_BANDS, PRETRAINED_BANDS)
    
    # Should keep encoder and decoder prefixes
    assert any("encoder." in k for k in result.keys())
    assert any("decoder." in k for k in result.keys())


def test_checkpoint_filter_fn_mae_v1_format():
    """Test checkpoint_filter_fn_mae with V1 weight format."""
    from terratorch.models.backbones.prithvi_mae import PrithviMAE
    
    model = MagicMock(spec=PrithviMAE)
    model.encoder = MagicMock()
    model.encoder.pos_embed = torch.randn(1, 100, 768)
    model.encoder.temporal_encoding = False
    model.encoder.location_encoding = False
    model.decoder = MagicMock()
    model.decoder.decoder_pos_embed = torch.randn(1, 50, 512)
    model.state_dict.return_value = {"encoder.patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16)}
    
    state_dict = {
        "patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16),  # V1 format without encoder prefix
        "decoder_blocks.0.weight": torch.randn(512, 512),  # V1 format
        "mask_token": torch.randn(768),
    }
    
    result = checkpoint_filter_fn_mae(state_dict, model, PRETRAINED_BANDS, PRETRAINED_BANDS)
    
    # V1 format should be converted to have encoder/decoder prefixes
    assert any("encoder." in k for k in result.keys())
    assert any("decoder." in k for k in result.keys())


def test_checkpoint_filter_fn_vit_adapter_basic():
    """Test checkpoint_filter_fn_vit_adapter."""
    from terratorch.models.backbones.prithvi_vit_adapter import PrithviViTAdapter
    
    # Skip if PrithviViTAdapter is the FallbackClass (optional dependency not installed)
    if not hasattr(PrithviViTAdapter, 'extra_layers'):
        pytest.skip("MultiScaleDeformableAttention not available")
    
    model = MagicMock(spec=PrithviViTAdapter)
    model.pos_embed = torch.randn(1, 100, 768)
    model.temporal_encoding = False
    model.location_encoding = False
    model.state_dict.return_value = {
        "level_embed": torch.randn(4, 768),
        "spm": torch.randn(768, 768),
        "patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16),
    }
    
    state_dict = {
        "patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16),
        "blocks.0.weight": torch.randn(768, 768),
        "cls_token": torch.randn(1, 1, 768),  # Should be dropped
        "norm.weight": torch.randn(768),  # Should be dropped
    }
    
    result = checkpoint_filter_fn_vit_adapter(state_dict, model, PRETRAINED_BANDS, PRETRAINED_BANDS)
    
    assert "cls_token" not in result
    assert not any("norm." in k for k in result.keys())
    assert "level_embed" in result
    assert "spm" in result


# Test _create_prithvi function
def test_create_prithvi_basic():
    """Test _create_prithvi with basic parameters."""
    model = _create_prithvi(
        "prithvi_eo_v2_300",
        pretrained=False,
        model_bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
        num_frames=1,
        encoder_only=True,
    )
    
    assert model is not None
    assert hasattr(model, "model_bands")
    assert len(model.model_bands) == 3


def test_create_prithvi_with_invalid_vit_adapter_variant():
    """Test _create_prithvi raises error for invalid vit_adapter variant."""
    with pytest.raises(ValueError, match="ViT Adapter not available"):
        _create_prithvi(
            "prithvi_eo_tiny",  # This variant doesn't have adapter config
            pretrained=False,
            vit_adapter=True,
        )


def test_create_prithvi_with_pretrained_cfg_overlay_error():
    """Test _create_prithvi raises error for deprecated pretrained_cfg_overlay."""
    with pytest.raises(ValueError, match="pretrained_cfg_overlay was removed"):
        _create_prithvi(
            "prithvi_eo_v2_300",
            pretrained=False,
            pretrained_cfg_overlay={"some": "config"},
        )


def test_create_prithvi_vit_adapter_encoder_only_error():
    """Test _create_prithvi raises error when vit_adapter with encoder_only=False."""
    with pytest.raises(ValueError, match="only supports encoder_only=True"):
        _create_prithvi(
            "prithvi_eo_v2_300",
            pretrained=False,
            vit_adapter=True,
            encoder_only=False,
        )


def test_create_prithvi_vit_adapter_with_out_indices_error():
    """Test _create_prithvi raises error when vit_adapter with out_indices."""
    try:
        with pytest.raises(ValueError, match="out_indices should not be provided"):
            _create_prithvi(
                "prithvi_eo_v2_300",
                pretrained=False,
                vit_adapter=True,
                out_indices=[0, 1, 2],
            )
    except ImportError as e:
        if "MultiScaleDeformableAttention" in str(e):
            pytest.skip("MultiScaleDeformableAttention not available")
        raise


def test_create_prithvi_with_ckpt_path():
    """Test _create_prithvi with checkpoint path."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        # Create a dummy checkpoint
        torch.save({"patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16)}, tmp.name)
        tmp_path = tmp.name
    
    try:
        model = _create_prithvi(
            "prithvi_eo_v2_300",
            pretrained=True,
            ckpt_path=tmp_path,
            model_bands=PRETRAINED_BANDS,
            num_frames=1,
        )
        assert model is not None
    finally:
        Path(tmp_path).unlink()


def test_create_prithvi_ckpt_path_without_pretrained_warning():
    """Test _create_prithvi warns when ckpt_path without pretrained."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save({}, tmp.name)
        tmp_path = tmp.name
    
    try:
        import logging
        with patch.object(logging.getLogger('terratorch.models.backbones.prithvi_vit'), 'warning') as mock_warning:
            model = _create_prithvi(
                "prithvi_eo_v2_300",
                pretrained=False,
                ckpt_path=tmp_path,
                num_frames=1,
            )
            mock_warning.assert_called()
    finally:
        Path(tmp_path).unlink()


def test_create_prithvi_mae_with_out_indices_error():
    """Test _create_prithvi raises assertion error for MAE with out_indices."""
    with pytest.raises(AssertionError, match="out_indices provided for a MAE model"):
        _create_prithvi(
            "prithvi_eo_v2_300",
            pretrained=False,
            encoder_only=False,
            out_indices=[0, 1],
        )


def test_create_prithvi_with_custom_out_indices():
    """Test _create_prithvi with custom out_indices."""
    model = _create_prithvi(
        "prithvi_eo_v2_300",
        pretrained=False,
        num_frames=1,
        encoder_only=True,
        out_indices=[0, 5, 11, 23],
    )
    
    assert hasattr(model, "out_indices")
    assert model.out_indices == [0, 5, 11, 23]


# Test registered model functions
def test_prithvi_eo_tiny():
    """Test prithvi_eo_tiny function."""
    model = prithvi_eo_tiny(pretrained=False, bands=[HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE])
    assert model is not None


def test_prithvi_eo_v1_100():
    """Test prithvi_eo_v1_100 function."""
    model = prithvi_eo_v1_100(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_tiny_tl():
    """Test prithvi_eo_v2_tiny_tl function."""
    model = prithvi_eo_v2_tiny_tl(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_100_tl():
    """Test prithvi_eo_v2_100_tl function."""
    model = prithvi_eo_v2_100_tl(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_300():
    """Test prithvi_eo_v2_300 function."""
    model = prithvi_eo_v2_300(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_600():
    """Test prithvi_eo_v2_600 function."""
    model = prithvi_eo_v2_600(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_300_tl():
    """Test prithvi_eo_v2_300_tl function."""
    model = prithvi_eo_v2_300_tl(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_600_tl():
    """Test prithvi_eo_v2_600_tl function."""
    model = prithvi_eo_v2_600_tl(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v1_100_mae():
    """Test prithvi_eo_v1_100_mae function."""
    model = prithvi_eo_v1_100_mae(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v1_100_mae_encoder_only_error():
    """Test prithvi_eo_v1_100_mae raises error with encoder_only=True."""
    with pytest.raises(ValueError, match="Please use 'prithvi_eo_v1_100' for encoder only"):
        prithvi_eo_v1_100_mae(pretrained=False, encoder_only=True)


def test_prithvi_eo_v2_tiny_tl_mae():
    """Test prithvi_eo_v2_tiny_tl_mae function."""
    model = prithvi_eo_v2_tiny_tl_mae(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_tiny_tl_mae_encoder_only_error():
    """Test prithvi_eo_v2_tiny_tl_mae raises error with encoder_only=True."""
    with pytest.raises(ValueError, match="Please use 'prithvi_eo_v2_tiny_tl' for encoder only"):
        prithvi_eo_v2_tiny_tl_mae(pretrained=False, encoder_only=True)


def test_prithvi_eo_v2_100_tl_mae():
    """Test prithvi_eo_v2_100_tl_mae function."""
    model = prithvi_eo_v2_100_tl_mae(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_100_tl_mae_encoder_only_error():
    """Test prithvi_eo_v2_100_tl_mae raises error with encoder_only=True."""
    with pytest.raises(ValueError, match="Please use 'prithvi_eo_v2_100_tl' for encoder only"):
        prithvi_eo_v2_100_tl_mae(pretrained=False, encoder_only=True)


def test_prithvi_eo_v2_300_mae():
    """Test prithvi_eo_v2_300_mae function."""
    model = prithvi_eo_v2_300_mae(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_300_mae_encoder_only_error():
    """Test prithvi_eo_v2_300_mae raises error with encoder_only=True."""
    with pytest.raises(ValueError, match="Please use 'prithvi_eo_v2_300' for encoder only"):
        prithvi_eo_v2_300_mae(pretrained=False, encoder_only=True)


def test_prithvi_eo_v2_300_tl_mae():
    """Test prithvi_eo_v2_300_tl_mae function."""
    model = prithvi_eo_v2_300_tl_mae(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_300_tl_mae_encoder_only_error():
    """Test prithvi_eo_v2_300_tl_mae raises error with encoder_only=True."""
    with pytest.raises(ValueError, match="Please use 'prithvi_eo_v2_300_tl' for encoder only"):
        prithvi_eo_v2_300_tl_mae(pretrained=False, encoder_only=True)


def test_prithvi_eo_v2_600_mae():
    """Test prithvi_eo_v2_600_mae function."""
    model = prithvi_eo_v2_600_mae(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_600_mae_encoder_only_error():
    """Test prithvi_eo_v2_600_mae raises error with encoder_only=True."""
    with pytest.raises(ValueError, match="Please use 'prithvi_eo_v2_600' for encoder only"):
        prithvi_eo_v2_600_mae(pretrained=False, encoder_only=True)


def test_prithvi_eo_v2_600_tl_mae():
    """Test prithvi_eo_v2_600_tl_mae function."""
    model = prithvi_eo_v2_600_tl_mae(pretrained=False, bands=PRETRAINED_BANDS)
    assert model is not None


def test_prithvi_eo_v2_600_tl_mae_encoder_only_error():
    """Test prithvi_eo_v2_600_tl_mae raises error with encoder_only=True."""
    with pytest.raises(ValueError, match="Please use 'prithvi_eo_v2_600_tl' for encoder only"):
        prithvi_eo_v2_600_tl_mae(pretrained=False, encoder_only=True)


# Test with vit_adapter
def test_prithvi_eo_v1_100_with_vit_adapter():
    """Test prithvi_eo_v1_100 with vit_adapter=True."""
    try:
        model = prithvi_eo_v1_100(pretrained=False, bands=PRETRAINED_BANDS, vit_adapter=True)
        assert model is not None
        assert hasattr(model, "out_indices")
    except ImportError as e:
        if "MultiScaleDeformableAttention" in str(e):
            pytest.skip("MultiScaleDeformableAttention not available")
        raise


def test_prithvi_eo_v2_300_with_vit_adapter():
    """Test prithvi_eo_v2_300 with vit_adapter=True."""
    try:
        model = prithvi_eo_v2_300(pretrained=False, bands=PRETRAINED_BANDS, vit_adapter=True)
        assert model is not None
    except ImportError as e:
        if "MultiScaleDeformableAttention" in str(e):
            pytest.skip("MultiScaleDeformableAttention not available")
        raise


def test_prithvi_eo_v2_300_tl_with_vit_adapter():
    """Test prithvi_eo_v2_300_tl with vit_adapter=True."""
    try:
        model = prithvi_eo_v2_300_tl(pretrained=False, bands=PRETRAINED_BANDS, vit_adapter=True)
        assert model is not None
    except ImportError as e:
        if "MultiScaleDeformableAttention" in str(e):
            pytest.skip("MultiScaleDeformableAttention not available")
        raise


# Test model_bands parameter handling
def test_create_prithvi_model_bands_from_kwargs():
    """Test _create_prithvi uses model_bands from kwargs for MAE models."""
    model_bands = [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]
    model = _create_prithvi(
        "prithvi_eo_v2_300",
        pretrained=False,
        encoder_only=False,
        model_bands=model_bands,
    )
    
    assert model.model_bands == model_bands


def test_mae_functions_use_model_bands_kwarg():
    """Test MAE functions correctly handle bands parameter."""
    bands = [HLSBands.RED, HLSBands.GREEN]
    model = prithvi_eo_v2_300_mae(pretrained=False, bands=bands)
    assert len(model.model_bands) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
