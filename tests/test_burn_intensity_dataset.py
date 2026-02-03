"""Comprehensive tests for burn_intensity.py dataset to maximize coverage."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rioxarray
import torch
from xarray import DataArray

from terratorch.datasets.burn_intensity import BurnIntensityNonGeo


# Helper functions
def create_dummy_tif(path: Path, bands: int = 6, height: int = 64, width: int = 64):
    """Create a dummy GeoTIFF file with random data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a simple raster with rioxarray
    data = np.random.rand(bands, height, width).astype(np.float32)
    coords = {
        'band': list(range(1, bands + 1)),
        'y': np.linspace(1000, 1000 + height, height),
        'x': np.linspace(2000, 2000 + width, width)
    }
    da = DataArray(data, coords=coords, dims=['band', 'y', 'x'])
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.to_raster(str(path))


def create_dummy_mask(path: Path, height: int = 64, width: int = 64):
    """Create a dummy mask file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create mask with class labels 0-4
    data = np.random.randint(0, 5, size=(1, height, width)).astype(np.float32)
    coords = {
        'band': [1],
        'y': np.linspace(1000, 1000 + height, height),
        'x': np.linspace(2000, 2000 + width, width)
    }
    da = DataArray(data, coords=coords, dims=['band', 'y', 'x'])
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.to_raster(str(path))


@pytest.fixture
def burn_intensity_data():
    """Create a temporary BurnIntensity dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir)
        
        # Create CSV files
        csv_full = data_root / "BS_files_raw.csv"
        csv_limited = data_root / "BS_files_with_less_than_25_percent_zeros.csv"
        
        df = pd.DataFrame({
            "Case_Name": ["Case001", "Case002", "Case003"]
        })
        df.to_csv(csv_full, index=False)
        
        df_limited = pd.DataFrame({
            "Case_Name": ["Case001", "Case002"]
        })
        df_limited.to_csv(csv_limited, index=False)
        
        # Create split files
        train_file = data_root / "train.txt"
        val_file = data_root / "val.txt"
        
        with open(train_file, 'w') as f:
            f.write("HLS_Case001.tif\n")
            f.write("HLS_Case002.tif\n")
        
        with open(val_file, 'w') as f:
            f.write("HLS_Case003.tif\n")
        
        # Create directories for time steps
        for time_step in ["pre", "during", "post"]:
            (data_root / time_step).mkdir(parents=True, exist_ok=True)
        
        # Create image files for each time step
        for case in ["Case001", "Case002", "Case003"]:
            for time_step in ["pre", "during", "post"]:
                create_dummy_tif(data_root / time_step / f"HLS_{case}.tif")
            
            # Create mask files (only in pre directory)
            create_dummy_mask(data_root / "pre" / f"BS_{case}.tif")
        
        yield str(data_root)


# Test initialization
def test_init_default_params(burn_intensity_data):
    """Test dataset initialization with default parameters."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    assert dataset.split == "train"
    assert dataset.bands == BurnIntensityNonGeo.all_band_names
    assert len(dataset.samples) == 2  # Case001, Case002


def test_init_custom_bands(burn_intensity_data):
    """Test initialization with custom bands."""
    bands = ("RED", "GREEN", "BLUE")
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        bands=bands
    )
    
    assert dataset.bands == bands
    assert len(dataset.band_indices) == 3


def test_init_val_split(burn_intensity_data):
    """Test initialization with validation split."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="val")
    
    assert dataset.split == "val"
    assert len(dataset.samples) == 1  # Case003


def test_init_invalid_split(burn_intensity_data):
    """Test that invalid split raises ValueError."""
    with pytest.raises(ValueError, match="Incorrect split"):
        BurnIntensityNonGeo(data_root=burn_intensity_data, split="invalid")


def test_init_invalid_bands(burn_intensity_data):
    """Test that invalid bands raise ValueError."""
    with pytest.raises(ValueError):
        BurnIntensityNonGeo(
            data_root=burn_intensity_data,
            split="train",
            bands=("INVALID_BAND",)
        )


def test_init_limited_data(burn_intensity_data):
    """Test initialization with limited data (less than 25% zeros)."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        use_full_data=False
    )
    
    assert len(dataset.samples) == 2  # Only Case001, Case002 in limited CSV


def test_init_with_custom_transform(burn_intensity_data):
    """Test initialization with custom transform."""
    transform = A.Compose([A.HorizontalFlip(p=0.5)])
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        transform=transform
    )
    
    assert dataset.transform == transform


def test_init_with_metadata(burn_intensity_data):
    """Test initialization with metadata enabled."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        use_metadata=True
    )
    
    assert dataset.use_metadata is True


def test_init_no_data_replace(burn_intensity_data):
    """Test initialization with custom no_data_replace value."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        no_data_replace=999.0
    )
    
    assert dataset.no_data_replace == 999.0


def test_init_no_label_replace(burn_intensity_data):
    """Test initialization with custom no_label_replace value."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        no_label_replace=-999
    )
    
    assert dataset.no_label_replace == -999


# Test methods
def test_len(burn_intensity_data):
    """Test __len__ method."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    assert len(dataset) == 2


def test_extract_basename(burn_intensity_data):
    """Test _extract_basename method."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    basename = dataset._extract_basename("/path/to/HLS_Case001.tif")
    assert basename == "HLS_Case001"


def test_extract_casename(burn_intensity_data):
    """Test _extract_casename method."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    casename = dataset._extract_casename("HLS_Case001.tif")
    assert casename == "Case001"
    
    casename = dataset._extract_casename("BS_Case001.tif")
    assert casename == "Case001"


def test_getitem_basic(burn_intensity_data):
    """Test __getitem__ method with basic configuration."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    sample = dataset[0]
    
    assert "image" in sample
    assert "mask" in sample
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    # After default transform: (C, T, H, W)
    assert sample["image"].shape[0] == 6  # 6 bands
    assert sample["image"].shape[1] == 3  # 3 time steps


def test_getitem_rgb_bands(burn_intensity_data):
    """Test __getitem__ with RGB bands only."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        bands=("RED", "GREEN", "BLUE")
    )
    
    sample = dataset[0]
    
    # After default transform: (C, T, H, W)
    assert sample["image"].shape[0] == 3  # 3 RGB bands


def test_getitem_with_metadata(burn_intensity_data):
    """Test __getitem__ with metadata enabled."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        use_metadata=True
    )
    
    sample = dataset[0]
    
    assert "location_coords" in sample
    assert isinstance(sample["location_coords"], torch.Tensor)
    assert sample["location_coords"].shape == (2,)  # lat, lon


def test_getitem_mask_dtype(burn_intensity_data):
    """Test that mask is returned as long tensor."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    sample = dataset[0]
    
    assert sample["mask"].dtype == torch.int64


def test_load_file(burn_intensity_data):
    """Test _load_file method."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    image_path = Path(burn_intensity_data) / "pre" / "HLS_Case001.tif"
    data = dataset._load_file(image_path)
    
    assert isinstance(data, DataArray)


def test_load_file_with_nan_replace(burn_intensity_data):
    """Test _load_file with NaN replacement."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        no_data_replace=999.0
    )
    
    image_path = Path(burn_intensity_data) / "pre" / "HLS_Case001.tif"
    data = dataset._load_file(image_path, nan_replace=999.0)
    
    assert not np.isnan(data.to_numpy()).any()


def test_load_file_no_nan_replace(burn_intensity_data):
    """Test _load_file without NaN replacement."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    image_path = Path(burn_intensity_data) / "pre" / "HLS_Case001.tif"
    data = dataset._load_file(image_path, nan_replace=None)
    
    assert isinstance(data, DataArray)


def test_get_coords(burn_intensity_data):
    """Test _get_coords method."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    image_path = Path(burn_intensity_data) / "pre" / "HLS_Case001.tif"
    image_data = rioxarray.open_rasterio(image_path)
    
    coords = dataset._get_coords(image_data)
    
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (2,)  # lat, lon
    assert coords.dtype == torch.float32


# Test plotting
def test_plot_basic(burn_intensity_data):
    """Test plot method with basic sample."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    sample = dataset[0]
    
    # Convert to expected format for plotting
    sample["image"] = sample["image"].permute(3, 0, 1, 2)  # (C, T, H, W)
    
    fig = dataset.plot(sample)
    
    assert fig is not None
    plt.close(fig)


def test_plot_with_prediction(burn_intensity_data):
    """Test plot method with prediction."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    sample = dataset[0]
    sample["image"] = sample["image"].permute(3, 0, 1, 2)  # (C, T, H, W)
    sample["prediction"] = torch.randint(0, 5, sample["mask"].shape)
    
    fig = dataset.plot(sample)
    
    assert fig is not None
    plt.close(fig)


def test_plot_with_suptitle(burn_intensity_data):
    """Test plot method with custom suptitle."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    sample = dataset[0]
    sample["image"] = sample["image"].permute(3, 0, 1, 2)
    
    fig = dataset.plot(sample, suptitle="Test Title")
    
    assert fig is not None
    assert fig._suptitle is not None
    plt.close(fig)


def test_plot_with_custom_class_names(burn_intensity_data):
    """Test plot method with custom class names."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    sample = dataset[0]
    sample["image"] = sample["image"].permute(3, 0, 1, 2)
    sample["class_names"] = ["Class0", "Class1", "Class2", "Class3", "Class4"]
    
    fig = dataset.plot(sample)
    
    assert fig is not None
    plt.close(fig)


def test_plot_missing_rgb_bands(burn_intensity_data):
    """Test that plot raises error when RGB bands are missing."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        bands=("NIR", "SWIR_1", "SWIR_2")  # No RGB bands
    )
    
    sample = dataset[0]
    sample["image"] = sample["image"].permute(3, 0, 1, 2)
    
    with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
        dataset.plot(sample)


# Test constants and attributes
def test_class_constants():
    """Test that class constants are properly defined."""
    assert len(BurnIntensityNonGeo.all_band_names) == 6
    assert len(BurnIntensityNonGeo.rgb_bands) == 3
    assert len(BurnIntensityNonGeo.class_names) == 5
    assert BurnIntensityNonGeo.num_classes == 5
    assert len(BurnIntensityNonGeo.time_steps) == 3


def test_band_sets():
    """Test BAND_SETS dictionary."""
    assert "all" in BurnIntensityNonGeo.BAND_SETS
    assert "rgb" in BurnIntensityNonGeo.BAND_SETS
    assert BurnIntensityNonGeo.BAND_SETS["all"] == BurnIntensityNonGeo.all_band_names
    assert BurnIntensityNonGeo.BAND_SETS["rgb"] == BurnIntensityNonGeo.rgb_bands


def test_csv_files():
    """Test CSV_FILES dictionary."""
    assert "limited" in BurnIntensityNonGeo.CSV_FILES
    assert "full" in BurnIntensityNonGeo.CSV_FILES


def test_splits():
    """Test splits dictionary."""
    assert "train" in BurnIntensityNonGeo.splits
    assert "val" in BurnIntensityNonGeo.splits


# Test edge cases
def test_samples_structure(burn_intensity_data):
    """Test that samples have the correct structure."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    for sample_info in dataset.samples:
        assert "image_files" in sample_info
        assert "mask_file" in sample_info
        assert "casename" in sample_info
        assert len(sample_info["image_files"]) == 3  # pre, during, post


def test_filtering_by_casenames(burn_intensity_data):
    """Test that samples are correctly filtered by casenames in CSV."""
    # Remove Case001 from train.txt but keep it in CSV to test filtering
    train_file = Path(burn_intensity_data) / "train.txt"
    with open(train_file, 'w') as f:
        f.write("HLS_Case002.tif\n")
        f.write("HLS_NonExistent.tif\n")  # This should be filtered out
    
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    # Only Case002 should remain (NonExistent is not in CSV)
    assert len(dataset.samples) == 1
    assert dataset.samples[0]["casename"] == "Case002"


def test_image_stacking(burn_intensity_data):
    """Test that images are correctly stacked across time steps."""
    dataset = BurnIntensityNonGeo(data_root=burn_intensity_data, split="train")
    
    sample = dataset[0]
    
    # After default transform: (C, T, H, W) -> (6, 3, H, W)
    assert sample["image"].shape[1] == 3  # 3 time steps


def test_transform_none(burn_intensity_data):
    """Test that default transform is used when transform is None."""
    dataset = BurnIntensityNonGeo(
        data_root=burn_intensity_data,
        split="train",
        transform=None
    )
    
    assert dataset.transform is not None  # default_transform is used


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
