"""Comprehensive tests for quantize_lucid module.

Tests cover utility functions, codebook classes, and VectorQuantize module
to ensure maximum code coverage.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch, MagicMock

from terratorch.models.backbones.terramind.tokenizer.quantizers.quantize_lucid import (
    exists, default, noop, l2norm, log, uniform_init, gumbel_noise, gumbel_sample,
    ema_inplace, laplace_smoothing, sample_vectors, pad_shape, sample_multinomial,
    add_noise, kmeans, orthgonal_loss_fn,
    EuclideanCodebook, CosineSimCodebook, VectorQuantize
)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_exists_true(self):
        """Test exists with non-None value."""
        assert exists(5) is True
        assert exists("string") is True
        assert exists([]) is True
    
    def test_exists_false(self):
        """Test exists with None value."""
        assert exists(None) is False
    
    def test_default_with_value(self):
        """Test default returns value when not None."""
        assert default(5, 10) == 5
        assert default("test", "default") == "test"
    
    def test_default_with_none(self):
        """Test default returns default when None."""
        assert default(None, 10) == 10
        assert default(None, "default") == "default"
    
    def test_noop(self):
        """Test noop does nothing."""
        result = noop(1, 2, 3, a=4, b=5)
        assert result is None
    
    def test_l2norm(self):
        """Test L2 normalization."""
        t = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
        normalized = l2norm(t)
        
        # Check that vectors have unit norm
        norms = torch.norm(normalized, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-6)
    
    def test_log(self):
        """Test log with clamping."""
        t = torch.tensor([1.0, 0.1, 1e-30])
        result = log(t, eps=1e-20)
        
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_uniform_init(self):
        """Test uniform initialization."""
        tensor = uniform_init(10, 20)
        
        assert tensor.shape == (10, 20)
        assert tensor.dtype == torch.float32
    
    def test_gumbel_noise(self):
        """Test gumbel noise generation."""
        t = torch.randn(5, 10)
        noise = gumbel_noise(t)
        
        assert noise.shape == t.shape
        assert not torch.isnan(noise).any()
        assert not torch.isinf(noise).any()
    
    def test_gumbel_sample_zero_temperature(self):
        """Test gumbel sampling with zero temperature."""
        t = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        result = gumbel_sample(t, temperature=0)
        
        assert result.shape == (2,)
        assert result[0] == 2  # argmax of first row
        assert result[1] == 0  # argmax of second row
    
    def test_gumbel_sample_nonzero_temperature(self):
        """Test gumbel sampling with non-zero temperature."""
        t = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        result = gumbel_sample(t, temperature=1.0)
        
        assert result.shape == (2,)
        assert torch.all(result >= 0)
        assert torch.all(result < 3)
    
    def test_ema_inplace(self):
        """Test exponential moving average in place."""
        moving_avg = torch.tensor([1.0, 2.0, 3.0])
        new = torch.tensor([2.0, 3.0, 4.0])
        original = moving_avg.clone()
        
        ema_inplace(moving_avg, new, decay=0.9)
        
        # Check that moving_avg was modified
        assert not torch.equal(moving_avg, original)
        # Check approximate value
        expected = original * 0.9 + new * 0.1
        assert torch.allclose(moving_avg, expected)
    
    def test_laplace_smoothing(self):
        """Test laplace smoothing."""
        x = torch.tensor([10.0, 0.0, 5.0])
        result = laplace_smoothing(x, n_categories=3, eps=1e-5)
        
        # All values should be positive
        assert torch.all(result > 0)
        # Sum should be close to 1
        assert torch.allclose(result.sum(), torch.tensor(1.0))
    
    def test_sample_vectors_enough_samples(self):
        """Test sample_vectors when we have enough samples."""
        samples = torch.randn(100, 10)
        result = sample_vectors(samples, num=20)
        
        assert result.shape == (20, 10)
    
    def test_sample_vectors_not_enough_samples(self):
        """Test sample_vectors when we don't have enough samples."""
        samples = torch.randn(10, 5)
        result = sample_vectors(samples, num=20)
        
        assert result.shape == (20, 5)
    
    def test_pad_shape(self):
        """Test pad_shape utility."""
        shape = [10, 20, 30]
        result = pad_shape(shape, size=50, dim=1)
        
        assert result == [10, 50, 30]
    
    def test_sample_multinomial(self):
        """Test multinomial sampling."""
        probs = torch.tensor([0.2, 0.3, 0.5])
        result = sample_multinomial(total_count=100, probs=probs)
        
        assert result.shape == (3,)
        assert result.sum() == 100
        assert torch.all(result >= 0)
    
    def test_add_noise(self):
        """Test add_noise."""
        x = torch.randn(5, 10)
        result = add_noise(x, eps=0.1)
        
        assert result.shape == x.shape
        # With significant noise, result should differ from original
        assert not torch.allclose(result, x, atol=0.01)
    
    def test_kmeans_euclidean(self):
        """Test k-means with Euclidean distance."""
        samples = torch.randn(100, 10)
        means, bins = kmeans(samples, num_clusters=5, num_iters=3, use_cosine_sim=False)
        
        assert means.shape == (5, 10)
        assert bins.shape == (5,)
        assert bins.sum() == 100
    
    def test_kmeans_cosine(self):
        """Test k-means with cosine similarity."""
        samples = torch.randn(100, 10)
        means, bins = kmeans(samples, num_clusters=5, num_iters=3, use_cosine_sim=True)
        
        assert means.shape == (5, 10)
        assert bins.shape == (5,)
        # With cosine similarity, means should be normalized
        norms = torch.norm(means, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)
    
    def test_orthogonal_loss_fn(self):
        """Test orthogonal loss function."""
        t = torch.randn(10, 20)
        loss = orthgonal_loss_fn(t)
        
        assert loss.shape == ()
        assert loss >= 0


class TestEuclideanCodebook:
    """Test EuclideanCodebook class."""
    
    def test_initialization_uniform(self):
        """Test initialization with uniform init."""
        codebook = EuclideanCodebook(
            dim=64,
            codebook_size=128,
            kmeans_init=False
        )
        
        assert codebook.codebook_size == 128
        assert codebook.embed.shape == (128, 64)
        assert codebook.initted.item() == 1.0
    
    def test_initialization_kmeans(self):
        """Test initialization with k-means init."""
        codebook = EuclideanCodebook(
            dim=64,
            codebook_size=128,
            kmeans_init=True
        )
        
        assert codebook.initted.item() == 0.0
    
    def test_forward_basic(self):
        """Test forward pass."""
        codebook = EuclideanCodebook(
            dim=64,
            codebook_size=128,
            kmeans_init=False
        )
        
        x = torch.randn(4, 16, 64)
        quantize, embed_ind = codebook(x)
        
        assert quantize.shape == (4, 16, 64)
        assert embed_ind.shape == (4, 16)
        assert torch.all(embed_ind >= 0)
        assert torch.all(embed_ind < 128)
    
    def test_init_embed(self):
        """Test codebook initialization from data."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=True,
            kmeans_iters=2
        )
        
        data = torch.randn(100, 32)
        codebook.init_embed_(data)
        
        assert codebook.initted.item() == 1.0
        assert codebook.embed.shape == (10, 32)
    
    def test_replace_batch_random(self):
        """Test batch random replacement."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False
        )
        
        samples = torch.randn(50, 32)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[0:3] = True
        
        old_embed = codebook.embed.clone()
        codebook.replace_batch_random(samples, mask)
        
        # Check that masked codes were replaced
        assert not torch.equal(codebook.embed[mask], old_embed[mask])
    
    def test_replace_linde_buzo_gray(self):
        """Test Linde-Buzo-Gray replacement."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            code_replacement_policy='linde_buzo_gray'
        )
        
        # Set some cluster sizes
        codebook.cluster_size = torch.tensor([10, 5, 3, 8, 2, 1, 0, 0, 6, 4], dtype=torch.float)
        
        mask = torch.zeros(10, dtype=torch.bool)
        mask[6:8] = True
        
        old_embed = codebook.embed.clone()
        codebook.replace_linde_buzo_gray(mask)
        
        # Check that masked codes were replaced
        assert not torch.equal(codebook.embed[mask], old_embed[mask])
    
    def test_expire_codes_batch_random(self):
        """Test code expiration with batch random policy."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            threshold_ema_dead_code=2,
            code_replacement_policy='batch_random'
        )
        
        # Set some cluster sizes below threshold
        codebook.cluster_size = torch.tensor([10, 1, 3, 8, 0, 5, 2, 9, 6, 4], dtype=torch.float)
        
        batch_samples = torch.randn(20, 32)
        old_embed = codebook.embed.clone()
        
        codebook.expire_codes_(batch_samples)
        
        # Codes with cluster_size < 2 should be replaced
        expired = codebook.cluster_size < 2
        if expired.any():
            assert not torch.equal(codebook.embed[expired], old_embed[expired])
    
    def test_expire_codes_linde_buzo_gray(self):
        """Test code expiration with LBG policy."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            threshold_ema_dead_code=2,
            code_replacement_policy='linde_buzo_gray'
        )
        
        codebook.cluster_size = torch.tensor([10, 1, 3, 8, 0, 5, 2, 9, 6, 4], dtype=torch.float)
        
        batch_samples = torch.randn(20, 32)
        old_embed = codebook.embed.clone()
        
        codebook.expire_codes_(batch_samples)
        
        expired = codebook.cluster_size < 2
        if expired.any():
            assert not torch.equal(codebook.embed[expired], old_embed[expired])
    
    def test_expire_codes_invalid_policy(self):
        """Test that invalid policy raises error."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            threshold_ema_dead_code=2,
            code_replacement_policy='invalid_policy'
        )
        
        codebook.cluster_size = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float)
        batch_samples = torch.randn(20, 32)
        
        with pytest.raises(ValueError):
            codebook.expire_codes_(batch_samples)
    
    def test_training_mode(self):
        """Test forward in training mode."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False
        )
        
        codebook.train()
        x = torch.randn(4, 16, 32)
        
        quantize, embed_ind = codebook(x)
        
        assert quantize.shape == (4, 16, 32)
        assert embed_ind.shape == (4, 16)
    
    def test_learnable_codebook(self):
        """Test learnable codebook."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            learnable_codebook=True
        )
        
        assert isinstance(codebook.embed, nn.Parameter)
    
    def test_sample_codebook_temp(self):
        """Test with temperature sampling."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            sample_codebook_temp=0.5
        )
        
        x = torch.randn(4, 16, 32)
        quantize, embed_ind = codebook(x)
        
        assert quantize.shape == (4, 16, 32)


class TestCosineSimCodebook:
    """Test CosineSimCodebook class."""
    
    def test_initialization_uniform(self):
        """Test initialization with uniform init."""
        codebook = CosineSimCodebook(
            dim=64,
            codebook_size=128,
            kmeans_init=False
        )
        
        assert codebook.codebook_size == 128
        assert codebook.embed.shape == (128, 64)
        # Embed should be normalized
        norms = torch.norm(codebook.embed, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(128), atol=1e-5)
    
    def test_initialization_kmeans(self):
        """Test initialization with k-means init."""
        codebook = CosineSimCodebook(
            dim=64,
            codebook_size=128,
            kmeans_init=True
        )
        
        assert codebook.initted.item() == 0.0
    
    def test_forward_basic(self):
        """Test forward pass."""
        codebook = CosineSimCodebook(
            dim=64,
            codebook_size=128,
            kmeans_init=False
        )
        
        x = torch.randn(4, 16, 64)
        quantize, embed_ind = codebook(x)
        
        assert quantize.shape == (4, 16, 64)
        assert embed_ind.shape == (4, 16)
        assert torch.all(embed_ind >= 0)
        assert torch.all(embed_ind < 128)
    
    def test_init_embed(self):
        """Test codebook initialization from data."""
        codebook = CosineSimCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=True,
            kmeans_iters=2
        )
        
        data = torch.randn(100, 32)
        codebook.init_embed_(data)
        
        assert codebook.initted.item() == 1.0
        assert codebook.embed.shape == (10, 32)
    
    def test_training_mode(self):
        """Test forward in training mode."""
        codebook = CosineSimCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False
        )
        
        codebook.train()
        x = torch.randn(4, 16, 32)
        
        quantize, embed_ind = codebook(x)
        
        assert quantize.shape == (4, 16, 32)
        assert embed_ind.shape == (4, 16)
    
    def test_expire_codes(self):
        """Test code expiration."""
        codebook = CosineSimCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            threshold_ema_dead_code=2
        )
        
        codebook.cluster_size = torch.tensor([10, 1, 3, 8, 0, 5, 2, 9, 6, 4], dtype=torch.float)
        batch_samples = torch.randn(20, 32)
        
        old_embed = codebook.embed.clone()
        codebook.expire_codes_(batch_samples)
        
        expired = codebook.cluster_size < 2
        if expired.any():
            assert not torch.equal(codebook.embed[expired], old_embed[expired])


class TestVectorQuantize:
    """Test VectorQuantize module."""
    
    def test_initialization_euclidean(self):
        """Test initialization with Euclidean codebook."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256
        )
        
        assert vq.codebook_size == 256
        assert vq.codebook.shape == (256, 128)
    
    def test_initialization_cosine(self):
        """Test initialization with cosine similarity codebook."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            use_cosine_sim=True
        )
        
        assert vq.codebook_size == 256
        assert vq.codebook.shape == (256, 128)
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256
        )
        
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 8, 8)
        assert embed_ind.shape == (2, 8, 8)
        assert loss.numel() == 1  # Loss is a scalar tensor
    
    def test_with_projection(self):
        """Test with projection layers."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            codebook_dim=64
        )
        
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 8, 8)
    
    def test_multihead(self):
        """Test with multiple heads."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            heads=4
        )
        
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 8, 8)
        assert embed_ind.shape == (2, 4, 8, 8)
    
    def test_commitment_loss(self):
        """Test commitment loss computation."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            commitment_weight=1.0
        )
        
        vq.train()
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert loss > 0
    
    def test_orthogonal_reg(self):
        """Test orthogonal regularization."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            orthogonal_reg_weight=0.1
        )
        
        vq.train()
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert loss > 0
    
    def test_orthogonal_reg_active_codes_only(self):
        """Test orthogonal reg with active codes only."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            orthogonal_reg_weight=0.1,
            orthogonal_reg_active_codes_only=True
        )
        
        vq.train()
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert loss > 0
    
    def test_orthogonal_reg_max_codes(self):
        """Test orthogonal reg with max codes limit."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            orthogonal_reg_weight=0.1,
            orthogonal_reg_max_codes=50
        )
        
        vq.train()
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert loss > 0
    
    def test_norm_latents(self):
        """Test with latent normalization."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            norm_latents=True
        )
        
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 8, 8)
    
    def test_channel_last(self):
        """Test with channel_last format."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            channel_last=True,
            accept_image_fmap=False
        )
        
        x = torch.randn(2, 64, 128)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 64, 128)
    
    def test_no_image_fmap(self):
        """Test without image feature map."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            accept_image_fmap=False,
            channel_last=False
        )
        
        x = torch.randn(2, 128, 64)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 64)
    
    def test_indices_to_embedding(self):
        """Test converting indices to embeddings."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256
        )
        
        indices = torch.randint(0, 256, (2, 8, 8))
        embedding = vq.indices_to_embedding(indices)
        
        assert embedding.shape == (2, 128, 8, 8)
    
    def test_eval_mode(self):
        """Test in evaluation mode."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            commitment_weight=1.0
        )
        
        vq.eval()
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert loss == 0  # No loss in eval mode
    
    def test_kmeans_init(self):
        """Test with k-means initialization."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            kmeans_init=True,
            kmeans_iters=2
        )
        
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 8, 8)
    
    def test_sample_codebook_temp(self):
        """Test with codebook temperature sampling."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            sample_codebook_temp=0.5
        )
        
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 8, 8)


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_sample(self):
        """Test with single sample."""
        vq = VectorQuantize(dim=64, codebook_size=128)
        x = torch.randn(1, 64, 4, 4)
        
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (1, 64, 4, 4)
        assert embed_ind.shape == (1, 4, 4)
    
    def test_large_codebook(self):
        """Test with large codebook."""
        vq = VectorQuantize(dim=64, codebook_size=1024)
        x = torch.randn(2, 64, 8, 8)
        
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 64, 8, 8)
    
    def test_small_spatial_dims(self):
        """Test with small spatial dimensions."""
        vq = VectorQuantize(dim=128, codebook_size=256)
        x = torch.randn(2, 128, 2, 2)
        
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 2, 2)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        vq = VectorQuantize(
            dim=64,
            codebook_size=128,
            commitment_weight=1.0,
            orthogonal_reg_weight=0.1
        )
        
        vq.train()
        x = torch.randn(2, 64, 4, 4, requires_grad=True)
        
        quantize, loss, embed_ind = vq(x)
        loss.backward()
        
        assert x.grad is not None
    
    def test_zero_threshold_dead_code(self):
        """Test with zero threshold for dead code."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            threshold_ema_dead_code=0
        )
        
        codebook.cluster_size = torch.zeros(10)
        batch_samples = torch.randn(20, 32)
        
        # Should not raise error or replace codes
        codebook.expire_codes_(batch_samples)
    
    def test_all_codes_above_threshold(self):
        """Test when all codes are above threshold."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            threshold_ema_dead_code=2
        )
        
        # All cluster sizes above threshold
        codebook.cluster_size = torch.ones(10) * 10
        batch_samples = torch.randn(20, 32)
        
        old_embed = codebook.embed.clone()
        codebook.expire_codes_(batch_samples)
        
        # No codes should be replaced
        assert torch.equal(codebook.embed, old_embed)
    
    def test_multihead_with_projection(self):
        """Test multihead with projection."""
        vq = VectorQuantize(
            dim=128,
            codebook_size=256,
            codebook_dim=32,
            heads=4
        )
        
        x = torch.randn(2, 128, 8, 8)
        quantize, loss, embed_ind = vq(x)
        
        assert quantize.shape == (2, 128, 8, 8)
        assert embed_ind.shape == (2, 4, 8, 8)
    
    def test_cosine_sim_with_all_options(self):
        """Test CosineSimCodebook with all options."""
        codebook = CosineSimCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            decay=0.9,
            eps=1e-6,
            threshold_ema_dead_code=1,
            code_replacement_policy='linde_buzo_gray',
            learnable_codebook=True,
            sample_codebook_temp=0.3
        )
        
        codebook.train()
        x = torch.randn(4, 16, 32)
        
        quantize, embed_ind = codebook(x)
        assert quantize.shape == (4, 16, 32)
    
    def test_euclidean_codebook_ema_update(self):
        """Test EMA update during training."""
        codebook = EuclideanCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            decay=0.95
        )
        
        codebook.train()
        x = torch.randn(8, 20, 32)
        
        old_cluster_size = codebook.cluster_size.clone()
        old_embed = codebook.embed.clone()
        
        quantize, embed_ind = codebook(x)
        
        # Check that EMA was updated
        assert not torch.equal(codebook.cluster_size, old_cluster_size)
    
    def test_cosine_codebook_ema_update(self):
        """Test CosineSimCodebook EMA update during training."""
        codebook = CosineSimCodebook(
            dim=32,
            codebook_size=10,
            kmeans_init=False,
            decay=0.95
        )
        
        codebook.train()
        x = torch.randn(8, 20, 32)
        
        old_cluster_size = codebook.cluster_size.clone()
        
        quantize, embed_ind = codebook(x)
        
        # Check that EMA was updated
        assert not torch.equal(codebook.cluster_size, old_cluster_size)
