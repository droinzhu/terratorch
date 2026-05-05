"""
DFC2020 DataModule for TerraTorch.
S2 (13-band) + S1 (VV/VH) → 8-class land cover segmentation.
Uses GFM-Bench file structure: metadata.csv + train/val/test/{s2,s1,lc}/.
"""
from __future__ import annotations

import os
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import tifffile
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

# GFM-Bench DFC2020 IGBP → 8-class remapping
# Values >= 18 or 255 treated as ignore (-1)
_IGBP_TO_DFC = {
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0,   # Forest
    6: 1, 7: 1,                         # Shrubland
    8: -1, 9: -1,                       # Savanna (ignore)
    10: 2,                              # Grassland
    11: 3,                              # Wetlands
    12: 4, 14: 4,                       # Croplands
    13: 5,                              # Urban
    15: -1,                             # Snow/Ice (ignore)
    16: 6,                              # Barren
    17: 7,                              # Water
}

S2_MEAN = [1370.19, 1184.38, 1120.77, 1136.26, 1263.74, 1645.40,
           1846.87, 1762.60, 1972.62, 582.73, 14.77, 1732.16, 1247.92]
S2_STD  = [633.15, 650.28, 712.13, 965.23, 948.98, 1108.07,
           1258.36, 1233.15, 1364.39, 472.38, 14.31, 1310.37, 1087.60]
S1_MEAN = [-12.55, -20.19]
S1_STD  = [5.26, 5.91]


def _remap_label(label: np.ndarray) -> np.ndarray:
    out = np.full_like(label, -1, dtype=np.int64)
    for igbp, dfc in _IGBP_TO_DFC.items():
        out[label == igbp] = dfc
    return out


class DFC2020Dataset(Dataset):
    def __init__(self, root: str, split: str, transform=None,
                 s2_key: str = "S2", s1_key: str = "S1",
                 concat_modalities: bool = False,
                 single_modality: str | None = None):
        self.root = Path(root)
        meta = pd.read_csv(self.root / "metadata.csv")
        sub = meta[meta["split"] == split]
        # Filter to samples where all three files exist
        mask = sub["optical_path"].apply(lambda p: (self.root / p).exists())
        dropped = (~mask).sum()
        if dropped:
            import warnings
            warnings.warn(f"DFC2020 [{split}]: skipping {dropped} samples with missing optical files.")
        self.samples = sub[mask].reset_index(drop=True)
        self.transform = transform
        self.s2_key = s2_key
        self.s1_key = s1_key
        self.concat_modalities = concat_modalities
        self.single_modality = single_modality

        self.s2_mean = np.array(S2_MEAN, dtype=np.float32)
        self.s2_std  = np.array(S2_STD,  dtype=np.float32)
        self.s1_mean = np.array(S1_MEAN, dtype=np.float32)
        self.s1_std  = np.array(S1_STD,  dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        # Load HxWxC → normalize → HxWxC (albumentations convention)
        try:
            s2 = tifffile.imread(self.root / row["optical_path"]).astype(np.float32)
            s1 = tifffile.imread(self.root / row["radar_path"]).astype(np.float32)
        except Exception:
            # Corrupted file (partial extraction) — retry with a random valid index
            fallback = int(np.random.randint(len(self.samples)))
            return self[fallback]
        lc_raw = tifffile.imread(self.root / row["label_path"])

        # Label: first channel is IGBP class
        if lc_raw.ndim == 3:
            lc_raw = lc_raw[:, :, 0]
        label = _remap_label(lc_raw).astype(np.int64)  # must be int64 for CrossEntropyLoss

        # Normalize
        s2 = (s2 - self.s2_mean) / (self.s2_std + 1e-6)
        s1 = (s1 - self.s1_mean) / (self.s1_std + 1e-6)

        if self.transform:
            # albumentations expects HxWxC for images, HxW for mask
            result = self.transform(
                image=s2,
                image1=s1,
                mask=label,
            )
            s2    = result["image"]           # tensor CxHxW after ToTensorV2
            s1    = result["image1"]
            label = result["mask"].long()     # CrossEntropyLoss requires int64

        if self.concat_modalities:
            import torch as _torch
            image = _torch.cat([s2, s1], dim=0)   # [15, H, W]
        elif self.single_modality == "S2":
            image = s2
        elif self.single_modality == "S1":
            image = s1
        else:
            image = {self.s2_key: s2, self.s1_key: s1}

        return {
            "image": image,
            "mask": label,
            "filename": row["optical_path"],
        }


def _build_transform(target_size: int, augment: bool) -> A.Compose:
    ops = [A.Resize(height=target_size, width=target_size)]
    if augment:
        ops.append(A.D4())
    ops.append(ToTensorV2(transpose_mask=False))
    return A.Compose(
        ops,
        additional_targets={"image1": "image"},
    )


class DFC2020DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 224,
        s2_key: str = "S2",
        s1_key: str = "S1",
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        single_modality: str | None = None,
        concat_modalities: bool = False,
    ):
        super().__init__()
        self.data_root         = data_root
        self.batch_size        = batch_size
        self.num_workers       = num_workers
        self.image_size        = image_size
        self.s2_key            = s2_key
        self.s1_key            = s1_key
        self.max_train_samples = max_train_samples
        self.max_val_samples   = max_val_samples
        self.single_modality   = single_modality
        self.concat_modalities = concat_modalities  # True → concat S2+S1 into single tensor

    def _subsample(self, ds, max_n):
        if max_n and len(ds) > max_n:
            import torch
            from torch.utils.data import Subset
            idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))[:max_n].tolist()
            return Subset(ds, idx)
        return ds

    def setup(self, stage=None):
        kw = dict(s2_key=self.s2_key, s1_key=self.s1_key,
                  concat_modalities=self.concat_modalities,
                  single_modality=self.single_modality)
        self.train_ds = self._subsample(
            DFC2020Dataset(self.data_root, "train", _build_transform(self.image_size, augment=True), **kw),
            self.max_train_samples)
        self.val_ds = self._subsample(
            DFC2020Dataset(self.data_root, "val", _build_transform(self.image_size, augment=False), **kw),
            self.max_val_samples)
        self.test_ds = DFC2020Dataset(self.data_root, "test",
                                       _build_transform(self.image_size, augment=False), **kw)

    def plot(self, sample, split=None):
        """No-op plot method required by terratorch base_task."""
        return None

    # Aliases expected by terratorch base_task
    @property
    def train_dataset(self):
        return self.train_ds.dataset if hasattr(self.train_ds, "dataset") else self.train_ds

    @property
    def val_dataset(self):
        return self.val_ds.dataset if hasattr(self.val_ds, "dataset") else self.val_ds

    @property
    def test_dataset(self):
        return self.test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)
