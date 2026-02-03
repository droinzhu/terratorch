"""Visualization utilities for terratorch."""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Tuple
import numpy as np


def plot_boxes_labels(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    box_color: str = "red",
    box_linewidth: float = 2,
    font_size: int = 10,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
):

    """
    Plot an image with bounding boxes and labels.
    
    Args:
        image: Tensor of shape [C, H, W] or [H, W, C] with values in [0, 1] or [0, 255]
        boxes: Tensor of shape [N, 4] with boxes in xyxy format (x1, y1, x2, y2)
        labels: Optional tensor of shape [N] with class labels
        scores: Optional tensor of shape [N] with confidence scores
        class_names: Optional list of class names for label mapping
        figsize: Figure size as (width, height)
        box_color: Color for bounding boxes
        box_linewidth: Line width for bounding boxes
        font_size: Font size for labels
        show: Whether to call plt.show()
    
    Returns:
        matplotlib Figure object
    """
    # Convert image to numpy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        
        # Handle different image formats
        if image.dim() == 3:
            if image.shape[0] in [1, 3, 4]:  # [C, H, W]
                image = image.permute(1, 2, 0)
        
        image = image.numpy()
    
    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    # Handle grayscale
    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    
    # Convert boxes to numpy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    
    # Convert labels to numpy
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Convert scores to numpy
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    # Create figure and axis
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(image)

    
    # Plot each box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=box_linewidth,
            edgecolor=box_color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label if available
        if labels is not None:
            label_text = ""
            
            # Add class name or ID
            if class_names is not None and i < len(labels):
                label_idx = int(labels[i])
                if label_idx < len(class_names):
                    label_text = class_names[label_idx]
                else:
                    label_text = f"Class {label_idx}"
            elif i < len(labels):
                label_text = f"Class {int(labels[i])}"
            
            # Add score if available
            if scores is not None and i < len(scores):
                label_text += f" {scores[i]:.2f}"
            
            # Draw label background and text
            if label_text:
                ax.text(
                    x1, y1 - 5,
                    label_text,
                    color='white',
                    fontsize=font_size,
                    bbox=dict(facecolor=box_color, alpha=0.7, edgecolor='none', pad=2)
                )
    
    ax.axis('off')
    plt.tight_layout()
    
    if show and ax is not None:
        plt.show()

    
    return fig


def plot_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    mask_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
):
    """
    Plot an image with a mask overlay.
    
    Args:
        image: Tensor of shape [C, H, W] or [H, W, C] with values in [0, 1] or [0, 255]
        mask: Tensor of shape [H, W] with binary mask values
        alpha: Transparency for mask overlay (0=transparent, 1=opaque)
        mask_color: RGB color for mask as tuple (r, g, b) in [0, 1]
        figsize: Figure size as (width, height)
        show: Whether to call plt.show()
    
    Returns:
        matplotlib Figure object
    """
    # Convert image to numpy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        
        if image.dim() == 3:
            if image.shape[0] in [1, 3, 4]:  # [C, H, W]
                image = image.permute(1, 2, 0)
        
        image = image.numpy()
    
    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    # Convert mask to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Create colored mask overlay
    mask_overlay = np.zeros((*mask.shape, 3))
    mask_overlay[mask > 0] = mask_color
    
    # Overlay mask with transparency
    ax.imshow(mask_overlay, alpha=alpha)
    
    ax.axis('off')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_image_mask_boxes(
    image: torch.Tensor,
    mask: torch.Tensor,
    boxes: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    alpha: float = 0.4,
    mask_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    box_color: str = "yellow",
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
):
    """
    Plot an image with both mask overlay and bounding boxes.
    
    Args:
        image: Tensor of shape [C, H, W] or [H, W, C]
        mask: Tensor of shape [H, W] with binary mask values
        boxes: Optional tensor of shape [N, 4] with boxes in xyxy format
        labels: Optional tensor of shape [N] with class labels
        alpha: Transparency for mask overlay
        mask_color: RGB color for mask as tuple (r, g, b) in [0, 1]
        box_color: Color for bounding boxes
        figsize: Figure size as (width, height)
        show: Whether to call plt.show()
    
    Returns:
        matplotlib Figure object
    """
    # Convert image to numpy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        
        if image.dim() == 3:
            if image.shape[0] in [1, 3, 4]:  # [C, H, W]
                image = image.permute(1, 2, 0)
        
        image = image.numpy()
    
    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    # Convert mask to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Create colored mask overlay
    mask_overlay = np.zeros((*mask.shape, 3))
    mask_overlay[mask > 0] = mask_color
    
    # Overlay mask with transparency
    ax.imshow(mask_overlay, alpha=alpha)
    
    # Plot boxes if provided
    if boxes is not None:
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=box_color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            if labels is not None and i < len(labels):
                ax.text(
                    x1, y1 - 5,
                    f"Class {int(labels[i])}",
                    color='white',
                    fontsize=10,
                    bbox=dict(facecolor=box_color, alpha=0.7, edgecolor='none', pad=2)
                )
    
    ax.axis('off')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig
