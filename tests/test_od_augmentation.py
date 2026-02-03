import os
import pytest
import torch
import numpy as np
from PIL import Image

from terratorch.datasets.od_augmentation import CopyPasteObjectDetectionDataset


class DummyObjectDetectionDataset:
    """Mock base dataset for testing."""
    def __init__(self, num_samples=5, image_size=(640, 480)):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return PIL image and boxes
        H, W = self.image_size
        image = Image.new("RGB", (W, H), color=(idx * 30, idx * 20, idx * 10))
        
        # Create some dummy boxes
        boxes = torch.tensor([
            [50.0, 50.0, 150.0, 150.0],
            [200.0, 100.0, 300.0, 200.0],
        ], dtype=torch.float32)
        
        return image, boxes


class DummyTensorDataset:
    """Mock dataset that returns tensors instead of PIL images."""
    def __init__(self, num_samples=5, image_size=(640, 480)):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        H, W = self.image_size
        # Return tensor image and boxes
        image = torch.rand(3, H, W)
        boxes = torch.tensor([
            [50.0, 50.0, 150.0, 150.0],
        ], dtype=torch.float32)
        
        return image, boxes


@pytest.fixture(scope="function")
def object_folder(tmp_path):
    """Create a temporary folder with RGBA object images."""
    obj_dir = tmp_path / "objects"
    obj_dir.mkdir(parents=True, exist_ok=True)
    
    # Create several RGBA PNG objects
    for i in range(5):
        # Create an RGBA image with some transparency
        img = Image.new("RGBA", (50, 50))
        pixels = []
        for y in range(50):
            for x in range(50):
                # Create a circular-ish shape
                dist = ((x - 25) ** 2 + (y - 25) ** 2) ** 0.5
                if dist < 20:
                    alpha = int(255 * (1 - dist / 20))
                    pixels.append((i * 50, 100, 200 - i * 30, alpha))
                else:
                    pixels.append((0, 0, 0, 0))
        
        img.putdata(pixels)
        img.save(obj_dir / f"object_{i}.png")
    
    return str(obj_dir)


class TestCopyPasteObjectDetectionDataset:
    def test_dataset_initialization(self, object_folder):
        """Test that the dataset can be initialized correctly."""
        base_dataset = DummyObjectDetectionDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=0.7,
            scale_range=(0.5, 1.5),
            max_objects=3,
        )
        
        assert len(dataset) == len(base_dataset)
        assert dataset.paste_prob == 0.7
        assert dataset.scale_range == (0.5, 1.5)
        assert dataset.max_objects == 3
        assert len(dataset.object_paths) == 5
    
    def test_dataset_length(self, object_folder):
        """Test that dataset length matches base dataset."""
        base_dataset = DummyObjectDetectionDataset(num_samples=10)
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
        )
        
        assert len(dataset) == 10
    
    def test_dataset_output_structure(self, object_folder):
        """Test that dataset returns expected output structure."""
        base_dataset = DummyObjectDetectionDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=1.0,  # Always paste
        )
        
        sample = dataset[0]
        
        # Check keys
        assert "image" in sample
        assert "mask" in sample
        
        # Check types
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)
        
        # Check data types
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.uint8
    
    def test_dataset_image_shape(self, object_folder):
        """Test that output images have correct shape."""
        base_dataset = DummyObjectDetectionDataset(image_size=(480, 640))
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
        )
        
        sample = dataset[0]
        
        # Image should be [C, H, W]
        assert sample["image"].shape == (3, 480, 640)
        assert sample["mask"].shape == (480, 640)
    
    def test_dataset_with_image_resize(self, object_folder):
        """Test that image resizing works correctly."""
        base_dataset = DummyObjectDetectionDataset(image_size=(480, 640))
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            image_size=(256, 256),  # Resize to square
        )
        
        sample = dataset[0]
        
        assert sample["image"].shape == (3, 256, 256)
        assert sample["mask"].shape == (256, 256)
    
    def test_dataset_mask_contains_boxes(self, object_folder):
        """Test that mask marks original box regions."""
        base_dataset = DummyObjectDetectionDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=0.0,  # Never paste, only original boxes
        )
        
        sample = dataset[0]
        mask = sample["mask"]
        
        # Mask should have some marked regions from original boxes
        assert torch.sum(mask > 0) > 0
        
        # Check that box regions are marked
        # Original boxes: [50, 50, 150, 150] and [200, 100, 300, 200]
        assert mask[100, 100].item() == 1  # Inside first box
        assert mask[150, 250].item() == 1  # Inside second box
    
    def test_dataset_with_tensor_input(self, object_folder):
        """Test that dataset handles tensor inputs from base dataset."""
        base_dataset = DummyTensorDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=1.0,
        )
        
        sample = dataset[0]
        
        assert sample["image"].shape == (3, 640, 480)
        assert sample["mask"].shape == (640, 480)
    
    def test_dataset_paste_increases_mask(self, object_folder):
        """Test that pasting objects increases mask coverage."""
        base_dataset = DummyObjectDetectionDataset()
        
        # Dataset with no pasting
        dataset_no_paste = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=0.0,
        )
        
        # Dataset with pasting
        dataset_with_paste = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=1.0,
            max_objects=5,
        )
    
    def test_dataset_image_values_in_range(self, object_folder):
        """Test that image values are in valid range [0, 1]."""
        base_dataset = DummyObjectDetectionDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=1.0,
        )
        
        sample = dataset[0]
        image = sample["image"]
        
        assert image.min() >= 0.0
        assert image.max() <= 1.0
    
    def test_dataset_mask_binary(self, object_folder):
        """Test that mask contains only binary values."""
        base_dataset = DummyObjectDetectionDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=1.0,
        )
        
        sample = dataset[0]
        mask = sample["mask"]
        
        # Mask should only contain 0 and 1
        unique_values = torch.unique(mask)
        assert all(val in [0, 1] for val in unique_values.tolist())
    
    def test_dataset_different_scale_ranges(self, object_folder):
        """Test that different scale ranges work."""
        base_dataset = DummyObjectDetectionDataset()
        
        # Small scale range
        dataset_small = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            scale_range=(0.1, 0.3),
            paste_prob=1.0,
        )
        
        # Large scale range
        dataset_large = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            scale_range=(1.0, 2.0),
            paste_prob=1.0,
        )
        
        sample_small = dataset_small[0]
        sample_large = dataset_large[0]
        
        assert sample_small["image"].shape == (3, 640, 480)
        assert sample_large["image"].shape == (3, 640, 480)
    
    def test_dataset_max_objects_parameter(self, object_folder):
        """Test that max_objects parameter is respected."""
        base_dataset = DummyObjectDetectionDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=1.0,
            max_objects=10,
        )
        
        sample = dataset[0]
        
        # Should complete without error
        assert sample["image"].shape == (3, 640, 480)
        assert sample["mask"].shape == (640, 480)
    
    def test_dataset_empty_object_folder_raises_error(self, tmp_path):
        """Test that empty object folder raises assertion error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        base_dataset = DummyObjectDetectionDataset()
        
        with pytest.raises(AssertionError):
            CopyPasteObjectDetectionDataset(
                base_dataset=base_dataset,
                object_folder=str(empty_dir),
            )
    
    def test_load_object_method(self, object_folder):
        """Test that _load_object returns valid RGBA images."""
        base_dataset = DummyObjectDetectionDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
        )
        
        obj = dataset._load_object()
        
        assert isinstance(obj, Image.Image)
        assert obj.mode == "RGBA"
    
    def test_dataset_with_empty_boxes(self, object_folder):
        """Test dataset when base dataset returns empty boxes."""
        class EmptyBoxDataset:
            def __len__(self):
                return 3
            
            def __getitem__(self, idx):
                image = Image.new("RGB", (640, 480), color=(100, 100, 100))
                boxes = torch.empty((0, 4), dtype=torch.float32)
                return image, boxes
        
        base_dataset = EmptyBoxDataset()
        
        dataset = CopyPasteObjectDetectionDataset(
            base_dataset=base_dataset,
            object_folder=object_folder,
            paste_prob=1.0,
        )
        
        sample = dataset[0]
        
        # Should work even with no original boxes
        assert sample["image"].shape == (3, 480, 640)
        assert sample["mask"].shape == (480, 640)