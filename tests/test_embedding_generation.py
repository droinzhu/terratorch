import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import torch.nn as nn

from terratorch.models.utils import TemporalWrapper
from terratorch.tasks.embedding_generation import EmbeddingGenerationTask


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def mock_backbone_registry():
    with patch("terratorch.tasks.embedding_generation.BACKBONE_REGISTRY") as registry:
        registry.build.return_value = MagicMock(
            spec=nn.Module,
            name="mock_backbone_model",
        )
        yield registry


def _has_warning(ws, substr: str) -> bool:
    return any(substr in str(w.message) for w in ws)


def test_init_defaults_geotiff_warning(temp_dir):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        task = EmbeddingGenerationTask(model="dummy", output_dir=temp_dir)

    assert task.output_path == Path(temp_dir)
    assert task.layers == [-1]
    assert task.output_format == "tiff"
    assert task.embed_file_key == "filename"
    assert _has_warning(caught, "GeoTIFF selected; 2D token embeddings")

def test_init_unsupported_output_format_raises(temp_dir):
    with pytest.raises(ValueError, match="Unsupported output format"):
        EmbeddingGenerationTask(
            model="dummy",
            output_dir=temp_dir,
            output_format="npy",  # unsupported
        )

def test_init_pooling_parquet_only_has_cls_warning(temp_dir):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        task = EmbeddingGenerationTask(
            model="dummy",
            output_dir=temp_dir,
            output_format="tiff",
            has_cls=None,
            embedding_pooling="vit_mean",
        )

    assert task.output_format == "tiff"
    msgs = [str(w.message) for w in caught]
    assert any("No 'has_cls' provided; assuming CLS" in m for m in msgs)

def test_init_pooling_warnings(temp_dir):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        task = EmbeddingGenerationTask(
            model="dummy",
            output_dir=temp_dir,
            output_format="tiff",
            has_cls=None,
            embedding_pooling="vit_mean",
        )

    assert task.embedding_pooling == "vit_mean"
    assert _has_warning(caught, "No 'has_cls' provided")
    assert _has_warning(caught, "GeoTIFF output not recommended")


def test_infer_bt_tensor_and_dict():
    task = EmbeddingGenerationTask(model="dummy")

    x = torch.randn(2, 3, 224, 224)
    assert task.infer_BT(x) == (2, 1)

    x_dict = {"optical": torch.randn(2, 3, 4, 224, 224)}
    assert task.infer_BT(x_dict) == (2, 4)


def test_check_file_ids_valid_and_errors():
    task = EmbeddingGenerationTask(model="dummy")
    x_4d = torch.randn(2, 3, 224, 224)

    # valid tensor (B,)
    task.check_file_ids(torch.tensor([0, 1]), x_4d)

    # invalid shape
    with pytest.raises(ValueError, match="shape mismatch"):
        task.check_file_ids(torch.zeros(3), x_4d)

    # invalid type
    with pytest.raises(
        TypeError,
        match="must be a tensor/ndarray or a \\(nested\\) list/tuple",
    ):
        task.check_file_ids("bad", x_4d)

    # valid nested list temporal
    x_5d = torch.randn(2, 3, 3, 224, 224)
    file_ids = [["t1", "t2", "t3"], ["t4", "t5", "t6"]]
    task.check_file_ids(file_ids, x_5d)


def test_check_file_ids_invalid_inner_length():
    task = EmbeddingGenerationTask(model="dummy")
    x = torch.randn(2, 3, 4, 224, 224)  # T = 4
    file_ids = [["t1", "t2"], ["t3", "t4"]]  # inner length 2 != 4

    with pytest.raises(ValueError, match="inner length 4"):
        task.check_file_ids(file_ids, x)


def test_configure_models_temporal_wrapper():
    task = EmbeddingGenerationTask(
        model="dummy",
        temporal_cfg={"temporal_wrapper": True, "temporal_pooling": "mean"},
    )

    assert isinstance(task.model, TemporalWrapper)


def test_get_embeddings():
    with patch("terratorch.tasks.embedding_generation.BACKBONE_REGISTRY.build") as build:
        build.return_value = MagicMock(
            return_value=[
                torch.randn(2, 4),
                torch.randn(2, 5),
            ]
        )
        task = EmbeddingGenerationTask(model="dummy")

    embeddings, layers = task.get_embeddings(torch.randn(2, 3), [0, -1])
    assert layers == [0, 1]
    assert len(embeddings) == 2


def test_get_embeddings_model_failure():
    model = MagicMock()
    model.side_effect = Exception("forward fail")

    with patch("terratorch.tasks.embedding_generation.BACKBONE_REGISTRY.build") as build:
        build.return_value = model
        task = EmbeddingGenerationTask(model="dummy")

    with pytest.raises(RuntimeError, match="Model inference failed"):
        task.get_embeddings(torch.randn(2, 3), [-1])


def test_pull_metadata_full_and_empty():
    task = EmbeddingGenerationTask(model="dummy")

    data = {
        "file_id": "fid",
        "time_": "2023-01-01",
        "centre_lat": 1.0,
        "centre_lon": 2.0,
        "extra": "keep",
    }
    meta = task.pull_metadata(data)
    assert meta["file_id"] == "fid"
    assert meta["time"] == "2023-01-01"
    assert meta["center_lat"] == 1.0
    assert meta["center_lon"] == 2.0
    assert "extra" in data
    assert "file_id" not in data

    data2 = {"x": 1}
    assert task.pull_metadata(data2) == {}


def test_pool_embedding_keep_and_vit_mean():
    task = EmbeddingGenerationTask(model="dummy")

    emb = torch.randn(10, 4)
    assert torch.equal(task.pool_embedding(emb, None, None), emb)

    vit = torch.randn(197, 8)
    out = task.pool_embedding(vit, "vit_mean", has_cls=True)
    assert out.shape == (8,)
    assert torch.allclose(out, vit[1:, :].mean(dim=0))


def test_pool_embedding_cnn_mean_and_errors():
    task = EmbeddingGenerationTask(model="dummy")

    cnn = torch.randn(64, 7, 7)
    out = task.pool_embedding(cnn, "cnn_mean", has_cls=None)
    assert out.shape == (64,)

    with pytest.raises(ValueError, match="Expected 2D embedding for ViT pooling"):
        task.pool_embedding(cnn, "vit_mean", has_cls=True)

    vit2d = torch.randn(197, 8)
    with pytest.raises(ValueError, match="Expected 3D embedding for CNN pooling"):
        task.pool_embedding(vit2d, "cnn_mean", has_cls=None)

    with pytest.raises(ValueError, match="Unsupported pooling method"):
        task.pool_embedding(vit2d, "weird_pool", has_cls=None)


def test_pool_embedding_vit_cls_error_without_cls():
    task = EmbeddingGenerationTask(model="dummy")

    emb = torch.randn(196, 8)
    with pytest.raises(ValueError, match="Cannot use 'vit_cls' pooling"):
        task.pool_embedding(emb, "vit_cls", has_cls=False)

def test_pool_embedding_vit_mean_and_cls_logic(temp_dir):
    task = EmbeddingGenerationTask(
        model="dummy",
        output_dir=temp_dir,
        output_format="tiff",
    )

    # 5 tokens (incl. CLS) x 4 dims
    emb = torch.stack([
        torch.zeros(4),        # CLS
        torch.ones(4),         # token 1
        2 * torch.ones(4),     # token 2
        3 * torch.ones(4),     # token 3
        4 * torch.ones(4),     # token 4
    ], dim=0)

    # CLS pooling: take first token
    cls_vec = task.pool_embedding(emb, pooling="vit_cls", has_cls=True)
    assert cls_vec.shape == (4,)
    assert torch.allclose(cls_vec, torch.zeros(4))

    # mean pooling with CLS: should drop CLS and average remaining tokens
    mean_vec = task.pool_embedding(emb, pooling="vit_mean", has_cls=True)
    assert mean_vec.shape == (4,)
    assert torch.allclose(mean_vec, 2.5 * torch.ones(4))

    # mean pooling without CLS: use all tokens
    mean_no_cls = task.pool_embedding(emb, pooling="vit_mean", has_cls=False)
    assert torch.allclose(mean_no_cls, 2.0 * torch.ones(4))


def test_write_tiff_1d_and_2d_and_3d(temp_dir):
    task = EmbeddingGenerationTask(model="dummy", output_dir=temp_dir)

    task.write_tiff(torch.randn(16), "s.tif", {"time": "2023-01-01"}, Path(temp_dir))
    out1 = Path(temp_dir) / "s_embedding.tif"
    assert out1.exists()

    task.has_cls = True
    vit = torch.randn(197, 8)  # 196 = 14x14
    task.write_tiff(vit, "vit.tif", {}, Path(temp_dir))
    out2 = Path(temp_dir) / "vit_embedding.tif"
    with rasterio.open(out2) as src:
        assert src.count == 8
        assert src.height == 14
        assert src.width == 14

    cnn = torch.randn(4, 5, 6)
    task.write_tiff(cnn, "cnn.tif", {}, Path(temp_dir))
    out3 = Path(temp_dir) / "cnn_embedding.tif"
    with rasterio.open(out3) as src:
        assert src.count == 4
        assert src.height == 5
        assert src.width == 6


def test_write_tiff_vit_nonsquare_raises(temp_dir):
    task = EmbeddingGenerationTask(
        model="dummy",
        output_dir=temp_dir,
        has_cls=True,
    )
    emb = torch.randn(200, 8)  # 199 patches
    with pytest.raises(ValueError, match="Cannot reshape"):
        task.write_tiff(emb, "bad.tif", {}, Path(temp_dir))


def test_write_parquet_1d(temp_dir):
    task = EmbeddingGenerationTask(
        model="dummy",
        output_dir=temp_dir,
        output_format="parquet",
    )
    emb = torch.randn(8)
    meta = {"time": np.array("2023-01-01")}
    task.write_parquet(emb, "p", meta, Path(temp_dir))

    out = Path(temp_dir) / "p_embedding.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert len(df["embedding"][0]) == 8
    assert df["time"][0] == "2023-01-01"


def test_save_embeddings_tensor_dict_and_invalid(temp_dir):
    task = EmbeddingGenerationTask(model="dummy", output_dir=temp_dir)
    file_ids = ["f1", "f2"]

    emb = torch.randn(2, 4, 3, 3)
    with patch.object(task, "write_batch") as wb:
        task.save_embeddings(emb, file_ids, {}, layer=0)
        wb.assert_called_once()

    emb_dict = {
        "optical": torch.randn(2, 4, 3, 3),
        "radar": torch.randn(2, 2, 3, 3),
    }
    with patch.object(task, "write_batch") as wb2:
        task.save_embeddings(emb_dict, file_ids, {}, layer=1)
        assert wb2.call_count == 2

    with pytest.raises(TypeError, match="Unsupported embedding type"):
        task.save_embeddings("bad", ["f1"], {}, layer=0)


def test_predict_step_image_dict_filename_in_image(temp_dir):
    model = MagicMock()
    model.return_value = torch.randn(2, 4, 3, 3)
    with patch("terratorch.tasks.embedding_generation.BACKBONE_REGISTRY") as reg:
        reg.build.return_value = model
        task = EmbeddingGenerationTask(model="dummy", output_dir=temp_dir)
        with patch.object(task, "save_embeddings") as save:
            batch = {
                "image": {
                    "optical": torch.randn(2, 3, 224, 224),
                    "filename": ["f1", "f2"],
                    "time": ["2023-01-01", "2023-01-02"],
                }
            }
            task.predict_step(batch)
            save.assert_called_once()

def test_predict_step_batch_filename_and_metadata(temp_dir):
    model = MagicMock()
    model.return_value = torch.randn(2, 4, 3, 3)

    with patch("terratorch.tasks.embedding_generation.BACKBONE_REGISTRY") as reg:
        reg.build.return_value = model
        task = EmbeddingGenerationTask(model="dummy", output_dir=temp_dir)

    with patch.object(task, "save_embeddings") as save:
        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "filename": ["f1", "f2"],
            "metadata": {"time": ["2023-01-01", "2023-01-02"]},
        }
        task.predict_step(batch)
        save.assert_called_once()

def test_predict_step_missing_filename_raises(temp_dir):
    model = MagicMock()
    model.return_value = torch.randn(2, 4, 3, 3)

    with patch("terratorch.tasks.embedding_generation.BACKBONE_REGISTRY") as reg:
        reg.build.return_value = model
        task = EmbeddingGenerationTask(model="dummy", output_dir=temp_dir)

    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "time": ["2023-01-01", "2023-01-02"],
    }
    with pytest.raises(KeyError, match="not found in input dictionary"):
        task.predict_step(batch)
def test_write_batch_temporal_and_non_temporal_tiff(temp_dir, monkeypatch):
    task = EmbeddingGenerationTask(
        model="dummy",
        output_dir=temp_dir,
        output_format="tiff",
    )

    calls = []

    def fake_write_tiff(emb, filename, meta, dir_path):
        calls.append((filename, meta))

    monkeypatch.setattr(task, "write_tiff", fake_write_tiff)

    emb_t = torch.randn(2, 2, 1, 1, 1)
    file_ids_t = [["f00.tif", "f01.tif"], ["f10.tif", "f11.tif"]]
    meta_t = {"time": np.arange(4).reshape(2, 2)}

    task.write_batch(emb_t, file_ids_t, meta_t, Path(temp_dir))

    assert len(calls) == 4
    f, m = calls[1]
    assert f == "f01.tif"
    assert m["time"] == 1

    calls.clear()
    emb_b = torch.randn(2, 1, 1, 1)
    file_ids_b = ["g0.tif", "g1.tif"]
    meta_b = {"time": np.array([10, 11])}

    task.write_batch(emb_b, file_ids_b, meta_b, Path(temp_dir))

    assert len(calls) == 2
    f, m = calls[0]
    assert f == "g0.tif"
    assert m["time"] == 10

def test_write_tiff_drops_cls_and_reshapes(temp_dir, monkeypatch):
    task = EmbeddingGenerationTask(
        model="dummy",
        output_dir=temp_dir,
        output_format="tiff",
        has_cls=True,
    )

    recorded = {}

    class DummyDst:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def write(self, arr):
            recorded["arr_shape"] = arr.shape

        def update_tags(self, **tags):
            recorded["tags"] = tags

    def fake_open(*args, **kwargs):
        return DummyDst()

    monkeypatch.setattr(
        "terratorch.tasks.embedding_generation.rasterio.open",
        fake_open,
    )

    # 5 tokens, dim=4 -> drop CLS -> 4 tokens -> 2x2 grid
    emb = torch.randn(5, 4)
    task.write_tiff(
        embedding=emb,
        filename="foo.tif",
        metadata={"id": np.array([1])},
        dir_path=Path(temp_dir),
    )

    assert recorded["arr_shape"] == (4, 2, 2)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])