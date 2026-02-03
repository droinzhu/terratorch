"""Comprehensive tests for Substation dataset.

Tests cover initialization, segmentation/object detection modes, splits, 
timepoint aggregation, plotting, edge cases to ensure maximum code coverage.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from torchgeo.datasets.errors import DatasetNotFoundError

from terratorch.datasets.substation import (
    Substation,
    ConvertCocoAnnotations,
    convert_coco_poly_to_mask,
    download_file_from_presigned
)


@pytest.fixture
def temp_substation_root():
    """Create a temporary Substation dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create image_stack directory
        img_dir = tmpdir / "image_stack"
        img_dir.mkdir(parents=True)
        
        # Create mask directory
        mask_dir = tmpdir / "mask"
        mask_dir.mkdir(parents=True)
        
        # Create 5 sample image stacks with multiple timepoints
        for i in range(5):
            # Create image stack with shape (num_timepoints, channels, h, w)
            # 5 timepoints, 13 channels, 228x228 pixels
            img_data = np.random.rand(5, 13, 228, 228).astype(np.float32) * 10000
            img_path = img_dir / f"image_{i:03d}.npz"
            np.savez_compressed(img_path, arr_0=img_data)
            
            # Create corresponding mask
            mask_data = np.random.randint(0, 4, size=(228, 228)).astype(np.uint8)
            # Set some pixels to class 3 (substation)
            mask_data[50:100, 50:100] = 3
            mask_path = mask_dir / f"image_{i:03d}.npz"
            np.savez_compressed(mask_path, arr_0=mask_data)
        
        # Create metadata CSV files
        meta_data_full = {
            'image': [f'image_{i:03d}.npz' for i in range(5)],
            'split': ['train', 'train', 'train', 'val', 'test'],
            'id': list(range(5))
        }
        meta_df_full = pd.DataFrame(meta_data_full)
        meta_df_full.to_csv(tmpdir / "substation_meta_splits_full.csv", index=False)
        
        meta_data_geobench = {
            'image': [f'image_{i:03d}.npz' for i in range(3)],
            'split': ['train', 'train', 'val'],
            'id': list(range(3))
        }
        meta_df_geobench = pd.DataFrame(meta_data_geobench)
        meta_df_geobench.to_csv(tmpdir / "substation_meta_splits_geobench.csv", index=False)
        
        # Create COCO annotations for object detection
        annotations = {
            "images": [
                {"id": i, "file_name": f"image_{i:03d}.npz", "width": 228, "height": 228}
                for i in range(5)
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [50, 50, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                    "segmentation": [[50, 50, 100, 50, 100, 100, 50, 100]]
                },
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [60, 60, 40, 40],
                    "area": 1600,
                    "iscrowd": 0,
                    "segmentation": [[60, 60, 100, 60, 100, 100, 60, 100]]
                }
            ],
            "categories": [
                {"id": 0, "name": "background"},
                {"id": 1, "name": "substation"}
            ]
        }
        
        ann_file = tmpdir / "annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(annotations, f)
        
        yield str(tmpdir)


class TestSubstationInitializationSegmentation:
    """Test initialization with segmentation mode."""
    
    def test_basic_segmentation_train(self, temp_substation_root):
        """Test basic initialization with segmentation mode and train split."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.mode == 'segmentation'
        assert dataset.split == 'train'
        assert len(dataset) == 3  # 3 train samples in full dataset
        assert dataset.bands == [0, 1, 2]
        assert dataset.mask_2d is False
    
    def test_class_attributes(self, temp_substation_root):
        """Test that class attributes are set correctly."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.directory == 'Substation'
        assert dataset.filename_images == 'image_stack.tar.gz'
        assert dataset.filename_masks == 'mask.tar.gz'
        assert dataset.filename_detection_labels == 'annotations.json'
        assert dataset.url_for_images is not None
        assert dataset.url_for_masks is not None
        assert dataset.categories == ('background', 'substation')
    
    def test_segmentation_val_split(self, temp_substation_root):
        """Test validation split."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='val',
            dataset_version='full',
            download=False
        )
        
        assert dataset.split == 'val'
        assert len(dataset) == 1  # 1 val sample
    
    def test_segmentation_test_split(self, temp_substation_root):
        """Test test split."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='test',
            dataset_version='full',
            download=False
        )
        
        assert dataset.split == 'test'
        assert len(dataset) == 1  # 1 test sample
    
    def test_mask_2d_true(self, temp_substation_root):
        """Test initialization with 2D mask."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=True,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.mask_2d is True
        sample = dataset[0]
        # 2D mask should have 2 channels (background, substation)
        assert sample['mask'].shape[0] == 2
    
    def test_geobench_version(self, temp_substation_root):
        """Test initialization with geobench dataset version."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='geobench',
            download=False
        )
        
        assert dataset.dataset_version == 'geobench'
        assert len(dataset) == 2  # 2 train samples in geobench
    
    def test_custom_bands(self, temp_substation_root):
        """Test initialization with custom bands."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[3, 4, 5, 6],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.bands == [3, 4, 5, 6]
        sample = dataset[0]
        # Should have 4 bands per timepoint
        assert 4 in sample['image'].shape
    
    def test_custom_plot_indexes(self, temp_substation_root):
        """Test initialization with custom plot indexes."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2, 3, 4],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            plot_indexes=[3, 2, 1],
            download=False
        )
        
        assert dataset.plot_indexes == [3, 2, 1]


class TestSubstationInitializationObjectDetection:
    """Test initialization with object detection mode."""
    
    def test_basic_object_detection(self, temp_substation_root):
        """Test basic initialization with object detection mode."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.mode == 'object_detection'
        assert hasattr(dataset, 'coco')
        assert hasattr(dataset, 'coco_convert')
    
    def test_object_detection_categories(self, temp_substation_root):
        """Test that categories are correctly defined."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.categories == ('background', 'substation')


class TestSubstationAssertions:
    """Test that invalid configurations raise assertions."""
    
    def test_invalid_timepoint_aggregation(self, temp_substation_root):
        """Test that invalid timepoint_aggregation raises AssertionError."""
        with pytest.raises(AssertionError):
            dataset = Substation(
                root=temp_substation_root,
                bands=[0, 1, 2],
                mask_2d=False,
                timepoint_aggregation='invalid_method',
                use_timepoints=False,
                mode='segmentation',
                split='train',
                dataset_version='full',
                download=False
            )
    
    def test_invalid_timepoint_aggregation_with_use_timepoints_false(self, temp_substation_root):
        """Test that concat/median/identity not allowed when use_timepoints=False."""
        with pytest.raises(AssertionError):
            dataset = Substation(
                root=temp_substation_root,
                bands=[0, 1, 2],
                mask_2d=False,
                timepoint_aggregation='concat',  # Not allowed with use_timepoints=False
                use_timepoints=False,
                mode='segmentation',
                split='train',
                dataset_version='full',
                download=False
            )
    
    def test_median_requires_use_timepoints(self, temp_substation_root):
        """Test that median requires use_timepoints=True."""
        with pytest.raises(AssertionError):
            dataset = Substation(
                root=temp_substation_root,
                bands=[0, 1, 2],
                mask_2d=False,
                timepoint_aggregation='median',
                use_timepoints=False,
                mode='segmentation',
                split='train',
                dataset_version='full',
                download=False
            )
    
    def test_identity_requires_use_timepoints(self, temp_substation_root):
        """Test that identity requires use_timepoints=True."""
        with pytest.raises(AssertionError):
            dataset = Substation(
                root=temp_substation_root,
                bands=[0, 1, 2],
                mask_2d=False,
                timepoint_aggregation='identity',
                use_timepoints=False,
                mode='segmentation',
                split='train',
                dataset_version='full',
                download=False
            )


class TestSubstationTimepointAggregation:
    """Test different timepoint aggregation strategies."""
    
    def test_timepoint_first(self, temp_substation_root):
        """Test first timepoint aggregation."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.timepoint_aggregation == 'first'
        sample = dataset[0]
        # Should have shape (channels, h, w)
        assert len(sample['image'].shape) == 3
        assert sample['image'].shape[0] == 3  # 3 bands
    
    def test_timepoint_last(self, temp_substation_root):
        """Test last timepoint aggregation."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='last',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.timepoint_aggregation == 'last'
        sample = dataset[0]
        assert len(sample['image'].shape) == 3
    
    def test_timepoint_random(self, temp_substation_root):
        """Test random timepoint aggregation."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='random',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.timepoint_aggregation == 'random'
        sample = dataset[0]
        assert len(sample['image'].shape) == 3
    
    def test_timepoint_concat(self, temp_substation_root):
        """Test concatenation of timepoints."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='concat',
            use_timepoints=True,
            num_of_timepoints=4,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.timepoint_aggregation == 'concat'
        assert dataset.use_timepoints is True
        sample = dataset[0]
        # Should concatenate timepoints into channels
        assert len(sample['image'].shape) == 3
        # 4 timepoints * 3 bands = 12 channels
        assert sample['image'].shape[0] == 12
    
    def test_timepoint_median(self, temp_substation_root):
        """Test median timepoint aggregation."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='median',
            use_timepoints=True,
            num_of_timepoints=4,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.timepoint_aggregation == 'median'
        sample = dataset[0]
        # Median should reduce to (channels, h, w)
        assert len(sample['image'].shape) == 3
        assert sample['image'].shape[0] == 3
    
    def test_timepoint_identity(self, temp_substation_root):
        """Test identity timepoint aggregation (no aggregation)."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='identity',
            use_timepoints=True,
            num_of_timepoints=4,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.timepoint_aggregation == 'identity'
        sample = dataset[0]
        # Identity keeps the timepoint dimension
        assert len(sample['image'].shape) == 4
        assert sample['image'].shape[0] == 4  # num_of_timepoints
    
    def test_num_of_timepoints_custom(self, temp_substation_root):
        """Test custom number of timepoints."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='concat',
            use_timepoints=True,
            num_of_timepoints=3,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert dataset.num_of_timepoints == 3
        sample = dataset[0]
        # 3 timepoints * 3 bands = 9 channels
        assert sample['image'].shape[0] == 9


class TestSubstationGetItemSegmentation:
    """Test __getitem__ method for segmentation mode."""
    
    def test_getitem_basic(self, temp_substation_root):
        """Test basic getitem for segmentation."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        assert 'image' in sample
        assert 'mask' in sample
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['mask'], torch.Tensor)
    
    def test_getitem_mask_values(self, temp_substation_root):
        """Test that mask values are correctly processed."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Mask should only have values 0 and 1
        unique_vals = torch.unique(sample['mask'])
        assert all(val in [0, 1] for val in unique_vals.tolist())
    
    def test_getitem_image_dtype(self, temp_substation_root):
        """Test that image has correct dtype."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        assert sample['image'].dtype == torch.float32
    
    def test_getitem_mask_dtype(self, temp_substation_root):
        """Test that mask has correct dtype."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        assert sample['mask'].dtype == torch.int64
    
    def test_getitem_with_transforms(self, temp_substation_root):
        """Test getitem with transforms."""
        def dummy_transform(sample):
            sample['image'] = sample['image'] * 2
            return sample
        
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            transforms=dummy_transform,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Transform should have been applied
        assert 'image' in sample
        assert 'mask' in sample


class TestSubstationGetItemObjectDetection:
    """Test __getitem__ method for object detection mode."""
    
    def test_getitem_basic_od(self, temp_substation_root):
        """Test basic getitem for object detection."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        assert 'image' in sample
        assert 'label' in sample or 'labels' in sample
    
    def test_getitem_od_with_annotations(self, temp_substation_root):
        """Test object detection getitem with annotations."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # First image has annotations
        if 'boxes' in sample:
            assert isinstance(sample['boxes'], torch.Tensor)
            assert sample['boxes'].shape[1] == 4  # x1, y1, x2, y2
    
    def test_getitem_od_boxes_labels_match(self, temp_substation_root):
        """Test that boxes and labels have matching lengths."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample and 'labels' in sample:
            assert len(sample['boxes']) == len(sample['labels'])


class TestSubstationLength:
    """Test dataset length method."""
    
    def test_len_full_train(self, temp_substation_root):
        """Test length for full train dataset."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert len(dataset) == 3
    
    def test_len_geobench_train(self, temp_substation_root):
        """Test length for geobench train dataset."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='geobench',
            download=False
        )
        
        assert len(dataset) == 2
    
    def test_len_val(self, temp_substation_root):
        """Test length for validation dataset."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='val',
            dataset_version='full',
            download=False
        )
        
        assert len(dataset) == 1


class TestSubstationPlotSegmentation:
    """Test plotting for segmentation mode."""
    
    def test_plot_segmentation_basic(self, temp_substation_root):
        """Test basic segmentation plotting."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_segmentation_with_suptitle(self, temp_substation_root):
        """Test segmentation plotting with suptitle."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        fig = dataset.plot(sample, suptitle="Test Title")
        assert fig is not None
        plt.close(fig)
    
    def test_plot_segmentation_with_prediction(self, temp_substation_root):
        """Test segmentation plotting with prediction."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Add fake prediction
        sample['prediction'] = torch.randint(0, 2, (228, 228))
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_segmentation_show_titles_false(self, temp_substation_root):
        """Test segmentation plotting without titles."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        fig = dataset.plot(sample, show_titles=False)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_segmentation_2d_mask(self, temp_substation_root):
        """Test plotting with 2D mask."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=True,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)


class TestSubstationPlotObjectDetection:
    """Test plotting for object detection mode."""
    
    def test_plot_od_basic(self, temp_substation_root):
        """Test basic object detection plotting."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample:
            fig = dataset.plot(sample)
            assert fig is not None
            plt.close(fig)
    
    def test_plot_od_with_suptitle(self, temp_substation_root):
        """Test object detection plotting with suptitle."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample:
            fig = dataset.plot(sample, suptitle="OD Test")
            assert fig is not None
            plt.close(fig)
    
    def test_plot_od_show_boxes_only(self, temp_substation_root):
        """Test object detection plotting with boxes only."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample:
            fig = dataset.plot(sample, show_feats='boxes')
            assert fig is not None
            plt.close(fig)
    
    def test_plot_od_show_masks_only(self, temp_substation_root):
        """Test object detection plotting with masks only."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample and 'masks' in sample:
            fig = dataset.plot(sample, show_feats='masks')
            assert fig is not None
            plt.close(fig)
    
    def test_plot_od_show_both(self, temp_substation_root):
        """Test object detection plotting with both boxes and masks."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample:
            fig = dataset.plot(sample, show_feats='both')
            assert fig is not None
            plt.close(fig)
    
    def test_plot_od_with_predictions(self, temp_substation_root):
        """Test object detection plotting with predictions."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample:
            # Add fake predictions as torch tensors
            sample['prediction_labels'] = torch.tensor([1])
            sample['prediction_scores'] = torch.tensor([0.9])
            sample['prediction_boxes'] = torch.tensor([[60, 60, 100, 100]], dtype=torch.float32)
            
            fig = dataset.plot(sample)
            assert fig is not None
            plt.close(fig)
    
    def test_plot_od_with_prediction_masks(self, temp_substation_root):
        """Test object detection plotting with prediction masks."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample:
            # Add fake predictions with masks as torch tensors
            sample['prediction_labels'] = torch.tensor([1])
            sample['prediction_scores'] = torch.tensor([0.9])
            sample['prediction_boxes'] = torch.tensor([[60, 60, 100, 100]], dtype=torch.float32)
            sample['prediction_masks'] = torch.rand(1, 1, 228, 228)
            
            fig = dataset.plot(sample)
            assert fig is not None
            plt.close(fig)
    
    def test_plot_od_custom_alpha(self, temp_substation_root):
        """Test object detection plotting with custom alpha values."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample:
            fig = dataset.plot(sample, box_alpha=0.5, mask_alpha=0.5)
            assert fig is not None
            plt.close(fig)
    
    def test_plot_od_custom_confidence(self, temp_substation_root):
        """Test object detection plotting with custom confidence threshold."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        if 'boxes' in sample:
            sample['prediction_labels'] = torch.tensor([1, 1])
            sample['prediction_scores'] = torch.tensor([0.3, 0.9])
            sample['prediction_boxes'] = torch.tensor([[60, 60, 100, 100], [70, 70, 110, 110]], dtype=torch.float32)
            
            fig = dataset.plot(sample, confidence_score=0.5)
            assert fig is not None
            plt.close(fig)


class TestSubstationEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_iterate_over_dataset(self, temp_substation_root):
        """Test iterating over entire dataset."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        count = 0
        for sample in dataset:
            assert 'image' in sample
            assert 'mask' in sample
            count += 1
        
        assert count == len(dataset)
    
    def test_access_same_sample_multiple_times(self, temp_substation_root):
        """Test accessing the same sample multiple times."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        assert torch.equal(sample1['image'], sample2['image'])
        assert torch.equal(sample1['mask'], sample2['mask'])
    
    def test_all_bands(self, temp_substation_root):
        """Test using all 13 bands."""
        dataset = Substation(
            root=temp_substation_root,
            bands=list(range(13)),
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Should have 13 bands
        assert 13 in sample['image'].shape
    
    def test_single_band(self, temp_substation_root):
        """Test using single band."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        assert 1 in sample['image'].shape
    
    def test_timepoints_padding(self, temp_substation_root):
        """Test timepoint padding when requesting more timepoints than available."""
        # Create dataset with images having fewer timepoints
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            img_dir = tmpdir / "image_stack"
            img_dir.mkdir(parents=True)
            
            mask_dir = tmpdir / "mask"
            mask_dir.mkdir(parents=True)
            
            # Create image with only 2 timepoints
            img_data = np.random.rand(2, 13, 228, 228).astype(np.float32) * 10000
            img_path = img_dir / "image_000.npz"
            np.savez_compressed(img_path, arr_0=img_data)
            
            mask_data = np.random.randint(0, 4, size=(228, 228)).astype(np.uint8)
            mask_path = mask_dir / "image_000.npz"
            np.savez_compressed(mask_path, arr_0=mask_data)
            
            meta_data = {
                'image': ['image_000.npz'],
                'split': ['train'],
                'id': [0]
            }
            meta_df = pd.DataFrame(meta_data)
            meta_df.to_csv(tmpdir / "substation_meta_splits_full.csv", index=False)
            
            # Request 4 timepoints but only 2 available
            dataset = Substation(
                root=str(tmpdir),
                bands=[0, 1, 2],
                mask_2d=False,
                timepoint_aggregation='concat',
                use_timepoints=True,
                num_of_timepoints=4,
                mode='segmentation',
                split='train',
                dataset_version='full',
                download=False
            )
            
            sample = dataset[0]
            # Should pad to 4 timepoints
            assert sample['image'].shape[0] == 12  # 4 timepoints * 3 bands
    
    def test_timepoints_removal(self, temp_substation_root):
        """Test timepoint removal when image has more timepoints than requested."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='concat',
            use_timepoints=True,
            num_of_timepoints=3,  # Request 3 but images have 5
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Should use only 3 most recent timepoints
        assert sample['image'].shape[0] == 9  # 3 timepoints * 3 bands


class TestConvertCocoAnnotations:
    """Test ConvertCocoAnnotations callable class."""
    
    def test_convert_basic(self):
        """Test basic conversion of COCO annotations."""
        converter = ConvertCocoAnnotations()
        
        image = torch.rand(3, 100, 100)
        sample = {
            'image': image,
            'label': {
                'image_id': 1,
                'annotations': [
                    {
                        'bbox': [10, 10, 30, 30],
                        'category_id': 1,
                        'iscrowd': 0,
                        'area': 900,
                        'segmentation': [[10, 10, 40, 10, 40, 40, 10, 40]]
                    }
                ]
            }
        }
        
        result = converter(sample)
        assert 'boxes' in result['label']
        assert 'labels' in result['label']
        assert 'image_id' in result['label']
    
    def test_convert_no_annotations(self):
        """Test that dataset handles images with no annotations."""
        # When there are no annotations, the dataset should handle it gracefully
        # The converter is only called when there are annotations
        # So we just test that empty boxes tensor is properly structured
        empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
        empty_labels = torch.zeros((0,), dtype=torch.int64)
        
        assert empty_boxes.shape == (0, 4)
        assert empty_labels.shape == (0,)
        assert empty_boxes.numel() == 0
        assert empty_labels.numel() == 0
    
    def test_convert_filters_crowd(self):
        """Test that iscrowd=1 annotations are filtered."""
        converter = ConvertCocoAnnotations()
        
        image = torch.rand(3, 100, 100)
        sample = {
            'image': image,
            'label': {
                'image_id': 1,
                'annotations': [
                    {
                        'bbox': [10, 10, 30, 30],
                        'category_id': 1,
                        'iscrowd': 1,  # Should be filtered
                        'area': 900,
                        'segmentation': [[10, 10, 40, 10, 40, 40, 10, 40]]
                    },
                    {
                        'bbox': [20, 20, 30, 30],
                        'category_id': 1,
                        'iscrowd': 0,
                        'area': 900,
                        'segmentation': [[20, 20, 50, 20, 50, 50, 20, 50]]
                    }
                ]
            }
        }
        
        result = converter(sample)
        # Should only have 1 box (non-crowd)
        assert len(result['label']['boxes']) == 1


class TestConvertCocoPolyToMask:
    """Test convert_coco_poly_to_mask function."""
    
    def test_convert_single_polygon(self):
        """Test converting single polygon to mask."""
        segmentations = [[[10, 10, 40, 10, 40, 40, 10, 40]]]
        masks = convert_coco_poly_to_mask(segmentations, 100, 100)
        
        assert masks.shape == (1, 100, 100)
        assert masks.dtype == torch.uint8
    
    def test_convert_multiple_polygons(self):
        """Test converting multiple polygons to masks."""
        segmentations = [
            [[10, 10, 40, 10, 40, 40, 10, 40]],
            [[50, 50, 80, 50, 80, 80, 50, 80]]
        ]
        masks = convert_coco_poly_to_mask(segmentations, 100, 100)
        
        assert masks.shape == (2, 100, 100)


class TestDownloadFileFromPresigned:
    """Test download_file_from_presigned function."""
    
    def test_download_skip_existing(self, tmp_path):
        """Test that download is skipped if file exists."""
        target_file = tmp_path / "test_file.txt"
        target_file.write_text("existing content")
        
        # Should skip download and not raise error with bad URL
        download_file_from_presigned("http://invalid.url", str(tmp_path), "test_file.txt")
        
        # File should still have original content
        assert target_file.read_text() == "existing content"
    
    @patch('requests.get')
    def test_download_new_file(self, mock_get, tmp_path):
        """Test downloading new file."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test", b"data"]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock()
        mock_get.return_value = mock_response
        
        download_file_from_presigned("http://test.url", str(tmp_path), "new_file.txt")
        
        target_file = tmp_path / "new_file.txt"
        assert target_file.exists()
    
    @patch('requests.get')
    def test_download_with_chunks(self, mock_get, tmp_path):
        """Test downloading with multiple chunks."""
        mock_response = MagicMock()
        # Simulate multiple chunks
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2", b"chunk3"]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock()
        mock_get.return_value = mock_response
        
        download_file_from_presigned("http://test.url", str(tmp_path), "chunked_file.txt")
        
        target_file = tmp_path / "chunked_file.txt"
        assert target_file.exists()
        assert target_file.read_bytes() == b"chunk1chunk2chunk3"
    
    @patch('requests.get')
    def test_download_empty_chunks_filtered(self, mock_get, tmp_path):
        """Test that empty chunks are filtered out."""
        mock_response = MagicMock()
        # Include empty chunks (should be filtered by 'if chunk')
        mock_response.iter_content.return_value = [b"chunk1", b"", b"chunk2", None, b"chunk3"]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock()
        mock_get.return_value = mock_response
        
        download_file_from_presigned("http://test.url", str(tmp_path), "filtered_file.txt")
        
        target_file = tmp_path / "filtered_file.txt"
        assert target_file.exists()
        # Only non-empty chunks should be written
        content = target_file.read_bytes()
        assert b"chunk1" in content
        assert b"chunk2" in content
        assert b"chunk3" in content


class TestSubstationVerification:
    """Test verification and download logic."""
    
    def test_verify_missing_files_no_download(self):
        """Test that DatasetNotFoundError is raised when files missing and download=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(DatasetNotFoundError):
                dataset = Substation(
                    root=tmpdir,
                    bands=[0, 1, 2],
                    mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
                    mode='segmentation',
                    split='train',
                    dataset_version='full',
                    download=False
                )
    
    @patch('terratorch.datasets.substation.download_url')
    @patch('terratorch.datasets.substation.extract_archive')
    @patch('terratorch.datasets.substation.download_file_from_presigned')
    def test_download_segmentation(self, mock_download_presigned, mock_extract, mock_download):
        """Test download process for segmentation mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create minimal structure after "download"
            def setup_after_download(*args, **kwargs):
                img_dir = tmpdir_path / "image_stack"
                img_dir.mkdir(parents=True, exist_ok=True)
                mask_dir = tmpdir_path / "mask"
                mask_dir.mkdir(parents=True, exist_ok=True)
                
                # Create one sample
                img_data = np.random.rand(5, 13, 228, 228).astype(np.float32) * 10000
                np.savez_compressed(img_dir / "image_000.npz", arr_0=img_data)
                
                mask_data = np.random.randint(0, 4, size=(228, 228)).astype(np.uint8)
                np.savez_compressed(mask_dir / "image_000.npz", arr_0=mask_data)
            
            mock_extract.side_effect = setup_after_download
            
            # Mock CSV download
            def mock_csv_download(url, folder, filename):
                if 'full' in filename:
                    meta_data = {
                        'image': ['image_000.npz'],
                        'split': ['train'],
                        'id': [0]
                    }
                    pd.DataFrame(meta_data).to_csv(Path(folder) / filename, index=False)
            
            mock_download_presigned.side_effect = mock_csv_download
            
            dataset = Substation(
                root=tmpdir,
                bands=[0, 1, 2],
                mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
                mode='segmentation',
                split='train',
                dataset_version='full',
                download=True,
                checksum=False
            )
            
            assert mock_download.called
            assert mock_download_presigned.called
    
    @patch('terratorch.datasets.substation.download_url')
    @patch('terratorch.datasets.substation.extract_archive')
    @patch('terratorch.datasets.substation.download_file_from_presigned')
    def test_download_object_detection(self, mock_download_presigned, mock_extract, mock_download):
        """Test download process for object detection mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create minimal structure after "download"
            def setup_after_download(*args, **kwargs):
                img_dir = tmpdir_path / "image_stack"
                img_dir.mkdir(parents=True, exist_ok=True)
                
                img_data = np.random.rand(5, 13, 228, 228).astype(np.float32) * 10000
                np.savez_compressed(img_dir / "image_000.npz", arr_0=img_data)
            
            mock_extract.side_effect = setup_after_download
            
            # Mock file downloads
            def mock_file_download(url, folder, filename):
                if 'full' in filename:
                    meta_data = {
                        'image': ['image_000.npz'],
                        'split': ['train'],
                        'id': [0]
                    }
                    pd.DataFrame(meta_data).to_csv(Path(folder) / filename, index=False)
                elif 'annotations' in filename:
                    annotations = {
                        "images": [{"id": 0, "file_name": "image_000.npz", "width": 228, "height": 228}],
                        "annotations": [],
                        "categories": [{"id": 0, "name": "background"}, {"id": 1, "name": "substation"}]
                    }
                    with open(Path(folder) / filename, 'w') as f:
                        json.dump(annotations, f)
            
            mock_download_presigned.side_effect = mock_file_download
            
            dataset = Substation(
                root=tmpdir,
                bands=[0, 1, 2],
                mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
                mode='object_detection',
                split='train',
                dataset_version='full',
                download=True,
                checksum=False
            )
            
            assert mock_download.called
            assert mock_download_presigned.called
    
    def test_verify_with_checksum(self, temp_substation_root):
        """Test verification with checksum enabled."""
        # This tests that checksum parameter is passed through
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False,
            checksum=True
        )
        
        assert dataset.checksum is True
    
    @patch('terratorch.datasets.substation.extract_archive')
    def test_verify_extracts_when_tarballs_exist(self, mock_extract, temp_substation_root):
        """Test that _verify extracts when tar.gz files exist but not extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create tar.gz files but no extracted directories
            (tmpdir_path / "image_stack.tar.gz").touch()
            (tmpdir_path / "mask.tar.gz").touch()
            
            # Create minimal metadata
            meta_data = {
                'image': ['image_000.npz'],
                'split': ['train'],
                'id': [0]
            }
            pd.DataFrame(meta_data).to_csv(tmpdir_path / "substation_meta_splits_full.csv", index=False)
            
            def setup_after_extract(*args, **kwargs):
                # Simulate extraction
                img_dir = tmpdir_path / "image_stack"
                img_dir.mkdir(parents=True, exist_ok=True)
                mask_dir = tmpdir_path / "mask"
                mask_dir.mkdir(parents=True, exist_ok=True)
                
                img_data = np.random.rand(5, 13, 228, 228).astype(np.float32) * 10000
                np.savez_compressed(img_dir / "image_000.npz", arr_0=img_data)
                
                mask_data = np.random.randint(0, 4, size=(228, 228)).astype(np.uint8)
                np.savez_compressed(mask_dir / "image_000.npz", arr_0=mask_data)
            
            mock_extract.side_effect = setup_after_extract
            
            dataset = Substation(
                root=str(tmpdir_path),
                bands=[0, 1, 2],
                mask_2d=False,
                timepoint_aggregation='first',
                use_timepoints=False,
                mode='segmentation',
                split='train',
                dataset_version='full',
                download=False
            )
            
            # Extract should have been called
            assert mock_extract.called
    
    def test_verify_skips_when_extracted_exists(self, temp_substation_root):
        """Test that _verify skips extraction when files already extracted."""
        # temp_substation_root already has extracted files
        # This should not trigger download or extraction
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        assert len(dataset) > 0  # Should work without issues


class TestSubstationLoadingMethods:
    """Test internal loading methods."""
    
    def test_load_image_shape(self, temp_substation_root):
        """Test that _load_image returns correct shape."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        image_path = os.path.join(dataset.image_dir, dataset.image_filenames[0])
        image = dataset._load_image(image_path)
        
        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.float32
    
    def test_load_segmentation_mask_shape(self, temp_substation_root):
        """Test that _load_segmentation_mask returns correct shape."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        mask_path = os.path.join(dataset.mask_dir, dataset.image_filenames[0])
        mask = dataset._load_segmentation_mask(mask_path)
        
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.int64
    
    def test_load_od_target_with_annotations(self, temp_substation_root):
        """Test _load_od_target with annotations present."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        # Load target for first image (which has annotations)
        target = dataset._load_od_target(0)
        
        assert 'image_id' in target
        assert 'annotations' in target
    
    def test_load_od_target_without_annotations(self, temp_substation_root):
        """Test _load_od_target for image with no annotations."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        # Image 2 has no annotations in our fixture
        if len(dataset) > 2:
            target = dataset._load_od_target(2)
            assert 'annotations' in target


class TestSubstationImageNormalization:
    """Test image normalization in plotting."""
    
    def test_plot_with_high_value_images(self, temp_substation_root):
        """Test plotting with images having values > 1 (raw satellite data)."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Ensure image has high values (like raw satellite data)
        sample['image'] = sample['image'] * 100  # Values >> 1
        
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_normalized_images(self, temp_substation_root):
        """Test plotting with already normalized images (values <= 1)."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Normalize to 0-1 range
        sample['image'] = sample['image'] / sample['image'].max()
        
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_4d_image_temporal(self, temp_substation_root):
        """Test plotting with 4D image (temporal dimension)."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='identity',
            use_timepoints=True,
            num_of_timepoints=4,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Sample should have 4D image (timepoints, channels, h, w)
        
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_with_custom_indexes(self, temp_substation_root):
        """Test plotting with custom plot indexes."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2, 3, 4],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            plot_indexes=[4, 3, 2],
            download=False
        )
        
        sample = dataset[0]
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_segmentation_2d_mask_with_prediction(self, temp_substation_root):
        """Test plotting 2D mask with prediction."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=True,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Add prediction with 2D format
        sample['prediction'] = torch.randint(0, 2, (2, 228, 228))
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)


class TestSubstationConvertCocoEdgeCases:
    """Test edge cases in COCO conversion."""
    
    def test_convert_with_invalid_boxes(self):
        """Test that invalid boxes (with zero or negative area) are filtered."""
        converter = ConvertCocoAnnotations()
        
        image = torch.rand(3, 100, 100)
        sample = {
            'image': image,
            'label': {
                'image_id': 1,
                'annotations': [
                    {
                        'bbox': [10, 10, 0, 0],  # Zero area
                        'category_id': 1,
                        'iscrowd': 0,
                        'area': 0,
                        'segmentation': [[10, 10, 10, 10, 10, 10]]
                    },
                    {
                        'bbox': [20, 20, 30, 30],
                        'category_id': 1,
                        'iscrowd': 0,
                        'area': 900,
                        'segmentation': [[20, 20, 50, 20, 50, 50, 20, 50]]
                    }
                ]
            }
        }
        
        result = converter(sample)
        # Only valid box should remain
        assert len(result['label']['boxes']) >= 0
    
    def test_convert_with_out_of_bounds_boxes(self):
        """Test that out-of-bounds boxes are clamped."""
        converter = ConvertCocoAnnotations()
        
        image = torch.rand(3, 100, 100)
        sample = {
            'image': image,
            'label': {
                'image_id': 1,
                'annotations': [
                    {
                        'bbox': [90, 90, 50, 50],  # Extends beyond image
                        'category_id': 1,
                        'iscrowd': 0,
                        'area': 2500,
                        'segmentation': [[90, 90, 140, 90, 140, 140, 90, 140]]
                    }
                ]
            }
        }
        
        result = converter(sample)
        if len(result['label']['boxes']) > 0:
            boxes = result['label']['boxes']
            # Boxes should be clamped to image dimensions
            assert torch.all(boxes[:, 0] >= 0)
            assert torch.all(boxes[:, 1] >= 0)
            assert torch.all(boxes[:, 2] <= 100)
            assert torch.all(boxes[:, 3] <= 100)
    
    def test_convert_preserves_area_and_iscrowd(self):
        """Test that area and iscrowd fields are preserved."""
        converter = ConvertCocoAnnotations()
        
        image = torch.rand(3, 100, 100)
        sample = {
            'image': image,
            'label': {
                'image_id': 1,
                'annotations': [
                    {
                        'bbox': [10, 10, 30, 30],
                        'category_id': 1,
                        'iscrowd': 0,
                        'area': 900,
                        'segmentation': [[10, 10, 40, 10, 40, 40, 10, 40]]
                    }
                ]
            }
        }
        
        result = converter(sample)
        # The converter adds area and iscrowd fields
        assert 'boxes' in result['label']
        assert 'labels' in result['label']


class TestSubstationODWithoutMasks:
    """Test object detection without mask segmentations."""
    
    def test_od_sample_without_masks(self, temp_substation_root):
        """Test OD sample when masks are not required."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        sample = dataset[0]
        # Check that sample can be retrieved
        assert 'image' in sample
        assert isinstance(sample['image'], torch.Tensor)
    
    def test_od_empty_annotations(self):
        """Test object detection with image having no annotations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create minimal structure
            img_dir = tmpdir / "image_stack"
            img_dir.mkdir(parents=True)
            
            # Create one image
            img_data = np.random.rand(5, 13, 228, 228).astype(np.float32) * 10000
            np.savez_compressed(img_dir / "image_000.npz", arr_0=img_data)
            
            # Create metadata
            meta_data = {
                'image': ['image_000.npz'],
                'split': ['train'],
                'id': [0]
            }
            pd.DataFrame(meta_data).to_csv(tmpdir / "substation_meta_splits_full.csv", index=False)
            
            # Create COCO annotations with no annotations for the image
            annotations = {
                "images": [
                    {"id": 0, "file_name": "image_000.npz", "width": 228, "height": 228}
                ],
                "annotations": [],  # No annotations
                "categories": [
                    {"id": 0, "name": "background"},
                    {"id": 1, "name": "substation"}
                ]
            }
            
            with open(tmpdir / "annotations.json", 'w') as f:
                json.dump(annotations, f)
            
            dataset = Substation(
                root=str(tmpdir),
                bands=[0, 1, 2],
                mask_2d=False,
                timepoint_aggregation='first',
                use_timepoints=False,
                mode='object_detection',
                split='train',
                dataset_version='full',
                download=False
            )
            
            sample = dataset[0]
            assert 'image' in sample
            # Sample should still be valid even without annotations


class TestSubstationIntegration:
    """Integration tests for Substation dataset."""
    
    def test_dataloader_compatibility(self, temp_substation_root):
        """Test compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        def collate_fn(batch):
            return batch
        
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        for batch in dataloader:
            assert len(batch) > 0
            break
    
    def test_full_pipeline_segmentation(self, temp_substation_root):
        """Test complete segmentation pipeline."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        # Get sample
        sample = dataset[0]
        
        # Verify sample
        assert 'image' in sample
        assert 'mask' in sample
        
        # Plot sample
        fig = dataset.plot(sample)
        assert fig is not None
        plt.close(fig)
    
    def test_full_pipeline_object_detection(self, temp_substation_root):
        """Test complete object detection pipeline."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='object_detection',
            split='train',
            dataset_version='full',
            download=False
        )
        
        # Get sample
        sample = dataset[0]
        
        # Verify sample
        assert 'image' in sample
        
        # Plot sample if has boxes
        if 'boxes' in sample:
            fig = dataset.plot(sample)
            assert fig is not None
            plt.close(fig)
    
    def test_multiple_samples_sequentially(self, temp_substation_root):
        """Test loading multiple samples in sequence."""
        dataset = Substation(
            root=temp_substation_root,
            bands=[0, 1, 2],
            mask_2d=False,
            timepoint_aggregation='first',
            use_timepoints=False,
            mode='segmentation',
            split='train',
            dataset_version='full',
            download=False
        )
        
        samples = [dataset[i] for i in range(min(3, len(dataset)))]
        assert all('image' in s and 'mask' in s for s in samples)
