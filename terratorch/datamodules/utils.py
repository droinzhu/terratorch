# Copyright contributors to the Terratorch project

import re
from collections.abc import Callable, Iterable

import albumentations as A
import numpy as np
import torch
import logging
logger = logging.getLogger("terratorch")


def wrap_in_compose_is_list(transform_list):
    # set check shapes to false because of the multitemporal case
    return A.Compose(transform_list, is_check_shapes=False) if isinstance(transform_list, Iterable) else transform_list


def check_dataset_stackability(dataset, batch_size: int, max_checks: int | None = 100) -> bool:
    if max_checks is None or len(dataset) <= max_checks:
        random_indexes = np.arange(len(dataset))
    else:
        random_indexes = np.random.randint(low=0, high=len(dataset), size=max_checks)
    shapes = np.array([dataset[idx]["image"].shape for idx in random_indexes])

    if len(shapes) > 1:
        if np.array_equal(np.max(shapes, 0), np.min(shapes, 0)):
            return batch_size
        else:
            logger.warning(
                "The batch samples can't be stacked, since they don't have the same dimensions. Setting batch_size=1."
            )
            return 1
    else:
        return batch_size


def check_dataset_stackability_dict(dataset, batch_size: int, max_checks: int | None = 100) -> bool:
    """Check stackability with item['image'] being a dict."""
    if max_checks is None or len(dataset) <= max_checks:
        random_indexes = np.arange(len(dataset))
    else:
        random_indexes = np.random.randint(low=0, high=len(dataset), size=max_checks)

    shapes = {}
    for idx in random_indexes:
        for mod, value in dataset[idx]["image"].items():
            if mod in shapes:
                shapes[mod].append(value.shape)
            else:
                shapes[mod] = [value.shape]

    if all(np.array_equal(np.max(s, 0), np.min(s, 0)) for s in shapes.values()):
        return batch_size
    else:
        logger.warning(
            "The batch samples can't be stacked, since they don't have the same dimensions. Setting batch_size=1."
        )
        return 1





class Normalize(Callable):
    """
    Unified normalization class for both regular and temporal images.
    
    Handles normalization for images with shapes:
    - (B, C, H, W): Regular 4D images
    - (B, C, T, H, W): Temporal 5D images
    
    Means and stds can be:
    - Shape (C,): For regular images or to average over temporal dimension
    - Shape (C, T): For temporal statistics applied to 5D images
    
    Args:
        means: Mean values. Can be list, numpy array, or torch tensor.
               Shape (C,) or (C, T).
        stds: Standard deviation values. Same format as means.
        denormalize: If True, reverses normalization (image * stds + means).
                    If False, applies normalization ((image - means) / stds).
                    Defaults to False.
    
    Examples:
        >>> # Regular 4D image
        >>> means = [123.5, 128.2, 129.1]
        >>> stds = [50.0, 51.2, 52.3]
        >>> norm = Normalize(means, stds)
        >>> batch = {"image": torch.randn(2, 3, 256, 256)}  # (B,C,H,W)
        >>> normalized = norm(batch)
        
        >>> # Temporal 5D image
        >>> means = [[100, 101], [200, 201], [300, 301]]  # (C,T) = (3,2)
        >>> stds = [[10, 11], [20, 21], [30, 31]]
        >>> norm = Normalize(means, stds)
        >>> batch = {"image": torch.randn(2, 3, 2, 256, 256)}  # (B,C,T,H,W)
        >>> normalized = norm(batch)
        
        >>> # Denormalization
        >>> norm_denorm = Normalize(means, stds, denormalize=True)
        >>> restored = norm_denorm({"image": normalized["image"]})
    """
    
    def __init__(self, means, stds, denormalize: bool = False):
        super().__init__()
        
        # Convert to torch tensors for consistent handling
        self.means = torch.tensor(means) if not isinstance(means, torch.Tensor) else means.clone()
        self.stds = torch.tensor(stds) if not isinstance(stds, torch.Tensor) else stds.clone()
        self.denormalize = denormalize
    
    def __call__(self, batch):
        """
        Apply normalization to batch images.
        
        Args:
            batch: Dictionary with "image" key containing tensor to normalize.
        
        Returns:
            Dictionary with normalized "image" tensor.
        """
        image = batch["image"]
        device = image.device
        
        means_tensor = self.means.to(device)
        stds_tensor = self.stds.to(device)
        
        if len(image.shape) == 5:
            # Image shape: (B, C, T, H, W)
            if len(self.means.shape) == 2:
                # Means shape: (C, T) - use full temporal statistics
                # Reshape to (1, C, T, 1, 1) for broadcasting
                means = means_tensor.view(1, -1, 1, 1, 1)
                stds = stds_tensor.view(1, -1, 1, 1, 1)
            else:
                # Means shape: (C,) - replicate across temporal dimension
                # Reshape to (1, C, 1, 1, 1) for broadcasting
                means = means_tensor.view(1, -1, 1, 1, 1)
                stds = stds_tensor.view(1, -1, 1, 1, 1)
        
        elif len(image.shape) == 4:
            # Image shape: (B, C, H, W)
            if len(self.means.shape) == 2:
                # Means shape: (C, T) - average over temporal dimension
                # Reshape to (1, C, 1, 1) for broadcasting
                means = means_tensor.mean(dim=1).view(1, -1, 1, 1)
                stds = stds_tensor.mean(dim=1).view(1, -1, 1, 1)
            else:
                # Means shape: (C,)
                # Reshape to (1, C, 1, 1) for broadcasting
                means = means_tensor.view(1, -1, 1, 1)
                stds = stds_tensor.view(1, -1, 1, 1)
        
        else:
            msg = f"Expected image with 4 or 5 dimensions, got {len(image.shape)}"
            raise ValueError(msg)
        
        # Apply normalization or denormalization
        if self.denormalize:
            batch["image"] = image * stds + means
        else:
            batch["image"] = (image - means) / stds
        
        return batch
