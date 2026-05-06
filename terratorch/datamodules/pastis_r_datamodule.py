"""
PASTIS-R DataModule for TerraTorch.

PASTIS-R is a multi-temporal, multi-modal (Sentinel-2 + Sentinel-1) dataset for
agricultural parcel semantic segmentation. This module handles the raw PASTIS-R
directory layout:

    PASTIS-R/
    ├── DATA_S2/      S2_XXXXXX.npy   (T_s2, 10, H, W)  float32
    ├── DATA_S1A/     S1A_XXXXXX.npy  (T_s1,  3, H, W)  float32  [ascending]
    ├── DATA_S1D/     S1D_XXXXXX.npy  (T_s1,  3, H, W)  float32  [descending]
    ├── ANNOTATIONS/  TARGET_XXXXXX.npy (H, W, 3) uint8
    └── metadata.geojson   GeoJSON with "ID" and "Fold" properties

Semantic classes: 0=background, 1-18=crop types, 19=void (ignore_index).
Standard fold split: folds 1-4 = train, fold 5 = val/test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import lightning.pytorch as pl

# ---------------------------------------------------------------------------
# Per-band normalisation constants  (PASTIS-R community standard)
# ---------------------------------------------------------------------------

# Sentinel-2: 10 bands → B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
S2_MEAN: List[float] = [
    1160.5, 1006.5,  974.0, 1129.2, 1311.8,
    1780.3, 2008.1, 2022.3, 1545.0, 1291.8,
]
S2_STD: List[float] = [
    741.2,  773.5,  830.5, 1029.7, 1057.2,
    1131.1, 1232.2, 1174.2, 1168.3, 1070.5,
]

# Sentinel-1: 3 channels → VH, VV, VH/VV
S1_MEAN: List[float] = [-19.8, -12.5,  0.6]
S1_STD:  List[float] = [  4.5,   4.2,  3.0]

NUM_S2_BANDS  = 10
NUM_S1_BANDS  = 3   # per orbit direction
VOID_CLASS    = 19  # ignore index for CrossEntropyLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PASTISRDataset(Dataset):
    """
    PyTorch Dataset for PASTIS-R.

    Each sample returns a dictionary:
        {
            "image": {
                s2_key: Tensor[num_s2_frames, 10, H, W],
                s1_key: Tensor[num_s1_frames, C_s1, H, W],
            },
            "mask":    Tensor[H, W]           int64,
            "s2_dates": Tensor[num_s2_frames] float32,
            "s1_dates": Tensor[num_s1_frames] float32,
        }

    S1 channel count: 3 per orbit direction → 3 (one dir) or 6 (both dirs).
    """

    def __init__(
        self,
        root: Union[str, Path],
        folds: List[int],
        num_s2_frames: int = 10,
        num_s1_frames: int = 10,
        use_s1a: bool = True,
        use_s1d: bool = False,
        transform=None,
        augment: bool = False,
        s2_key: str = "S2",
        s1_key: str = "S1",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            root:           Path to PASTIS-R root directory.
            folds:          List of fold indices to include (1-5).
            num_s2_frames:  Number of Sentinel-2 temporal frames to subsample.
            num_s1_frames:  Number of Sentinel-1 temporal frames to subsample.
            use_s1a:        Use ascending Sentinel-1 (DATA_S1A).
            use_s1d:        Use descending Sentinel-1 (DATA_S1D).
            transform:      Optional callable applied to the final sample dict.
            augment:        Random horizontal + vertical flip augmentation.
            s2_key:         Key used for the Sentinel-2 tensor inside "image".
            s1_key:         Key used for the Sentinel-1 tensor inside "image".
            max_samples:    If set, randomly subsample to at most this many patches.
        """
        if not use_s1a and not use_s1d:
            raise ValueError("At least one of use_s1a or use_s1d must be True.")

        self.root         = Path(root)
        self.num_s2_frames = num_s2_frames
        self.num_s1_frames = num_s1_frames
        self.use_s1a      = use_s1a
        self.use_s1d      = use_s1d
        self.transform    = transform
        self.augment      = augment
        self.s2_key       = s2_key
        self.s1_key       = s1_key

        # Normalisation arrays: (C,) for broadcasting over (T, C, H, W)
        self._s2_mean = torch.tensor(S2_MEAN, dtype=torch.float32).view(1, NUM_S2_BANDS, 1, 1)
        self._s2_std  = torch.tensor(S2_STD,  dtype=torch.float32).view(1, NUM_S2_BANDS, 1, 1)

        num_s1_ch = NUM_S1_BANDS * (int(use_s1a) + int(use_s1d))
        self._s1_mean = torch.tensor(
            S1_MEAN * (int(use_s1a) + int(use_s1d)), dtype=torch.float32
        ).view(1, num_s1_ch, 1, 1)
        self._s1_std = torch.tensor(
            S1_STD  * (int(use_s1a) + int(use_s1d)), dtype=torch.float32
        ).view(1, num_s1_ch, 1, 1)

        # Load patch IDs from metadata.geojson
        meta_path = self.root / "metadata.geojson"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.geojson not found at {meta_path}")

        with open(meta_path, "r") as f:
            geojson = json.load(f)

        folds_set = set(folds)
        patch_ids = [
            feat["properties"]["ID"]
            for feat in geojson["features"]
            if feat["properties"]["Fold"] in folds_set
        ]

        # Filter to patches where required files actually exist on disk
        valid_ids = []
        for pid in patch_ids:
            if not self._s2_path(pid).exists():
                continue
            if not self._ann_path(pid).exists():
                continue
            if use_s1a and not self._s1a_path(pid).exists():
                continue
            if use_s1d and not self._s1d_path(pid).exists():
                continue
            valid_ids.append(pid)

        if max_samples is not None and len(valid_ids) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(valid_ids), size=max_samples, replace=False)
            valid_ids = [valid_ids[i] for i in sorted(indices)]

        self.patch_ids: List[int] = valid_ids

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _s2_path(self, pid: int) -> Path:
        return self.root / "DATA_S2"    / f"S2_{pid}.npy"

    def _s1a_path(self, pid: int) -> Path:
        return self.root / "DATA_S1A"   / f"S1A_{pid}.npy"

    def _s1d_path(self, pid: int) -> Path:
        return self.root / "DATA_S1D"   / f"S1D_{pid}.npy"

    def _ann_path(self, pid: int) -> Path:
        return self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy"

    # ------------------------------------------------------------------
    # Temporal subsampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _subsample_frames(array: np.ndarray, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniformly subsample `num_frames` frames from a (T, C, H, W) array.

        Returns:
            sampled_array:  (num_frames, C, H, W)
            frame_indices:  (num_frames,) int  in [0, T)
        """
        T = array.shape[0]
        if T == 0:
            raise ValueError("Array has 0 temporal frames.")
        if num_frames >= T:
            # Repeat last frame if T < num_frames
            indices = np.linspace(0, T - 1, num_frames, dtype=int)
        else:
            indices = np.linspace(0, T - 1, num_frames, dtype=int)
        return array[indices], indices

    @staticmethod
    def _indices_to_doy(indices: np.ndarray, T_total: int, max_doy: float = 365.0) -> np.ndarray:
        """
        Map frame indices [0, T_total) to day-of-year values in [0, max_doy].

        PASTIS-R does not include per-image date metadata in the .npy files.
        We use a uniform spacing as a placeholder that is consistent across
        the whole time series.
        """
        if T_total <= 1:
            return np.array([0.0] * len(indices), dtype=np.float32)
        return (indices / (T_total - 1) * max_doy).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.patch_ids[idx]

        # ── Sentinel-2  (T_s2, 10, H, W) ──────────────────────────────
        s2_raw = np.load(self._s2_path(pid)).astype(np.float32)   # (T, 10, H, W)
        T_s2 = s2_raw.shape[0]
        s2_frames, s2_indices = self._subsample_frames(s2_raw, self.num_s2_frames)
        s2_doy = self._indices_to_doy(s2_indices, T_s2)

        # ── Sentinel-1  (T_s1, 3, H, W) per direction ─────────────────
        s1_parts = []
        s1_ref_T = None
        s1_ref_indices = None

        if self.use_s1a:
            s1a_raw = np.load(self._s1a_path(pid)).astype(np.float32)  # (T, 3, H, W)
            T_s1a = s1a_raw.shape[0]
            s1a_frames, s1a_indices = self._subsample_frames(s1a_raw, self.num_s1_frames)
            s1_parts.append(s1a_frames)
            s1_ref_T = T_s1a
            s1_ref_indices = s1a_indices

        if self.use_s1d:
            s1d_raw = np.load(self._s1d_path(pid)).astype(np.float32)  # (T, 3, H, W)
            T_s1d = s1d_raw.shape[0]
            # Use the same number of frames; if s1a was loaded use its indices for consistency
            if s1_ref_indices is not None:
                # Clamp indices to valid range of s1d
                clamped = np.clip(s1_ref_indices, 0, T_s1d - 1)
                s1d_frames = s1d_raw[clamped]
            else:
                s1d_frames, s1d_indices = self._subsample_frames(s1d_raw, self.num_s1_frames)
                s1_ref_T = T_s1d
                s1_ref_indices = s1d_indices
            s1_parts.append(s1d_frames)

        # Concatenate along channel axis:  (T, 3) + (T, 3) → (T, 6)
        s1_frames = np.concatenate(s1_parts, axis=1)  # (num_s1_frames, C_s1, H, W)
        s1_doy = self._indices_to_doy(s1_ref_indices, s1_ref_T)

        # ── Semantic label  (H, W) ─────────────────────────────────────
        target_raw = np.load(self._ann_path(pid))     # (H, W, 3) uint8
        label = target_raw[:, :, 0].astype(np.int64)  # semantic channel

        # Pixels with class==19 are void/ignore; all other values in [0,19] are valid.

        # ── Convert to torch tensors ───────────────────────────────────
        s2_tensor = torch.from_numpy(s2_frames)   # (num_s2_frames, 10, H, W)
        s1_tensor = torch.from_numpy(s1_frames)   # (num_s1_frames, C_s1, H, W)
        mask      = torch.from_numpy(label)       # (H, W)  int64
        s2_dates  = torch.from_numpy(s2_doy)      # (num_s2_frames,) float32
        s1_dates  = torch.from_numpy(s1_doy)      # (num_s1_frames,) float32

        # ── Augmentation (consistent across all temporal frames) ───────
        if self.augment:
            if torch.rand(1).item() > 0.5:
                s2_tensor = torch.flip(s2_tensor, dims=[-2])
                s1_tensor = torch.flip(s1_tensor, dims=[-2])
                mask      = torch.flip(mask,      dims=[-2])
            if torch.rand(1).item() > 0.5:
                s2_tensor = torch.flip(s2_tensor, dims=[-1])
                s1_tensor = torch.flip(s1_tensor, dims=[-1])
                mask      = torch.flip(mask,      dims=[-1])

        # ── Normalisation ──────────────────────────────────────────────
        s2_tensor = (s2_tensor - self._s2_mean) / (self._s2_std + 1e-6)
        s1_tensor = (s1_tensor - self._s1_mean) / (self._s1_std + 1e-6)

        sample = {
            "image": {
                self.s2_key: s2_tensor,
                self.s1_key: s1_tensor,
            },
            "mask":     mask,
            "s2_dates": s2_dates,
            "s1_dates": s1_dates,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class PASTISRDataModule(pl.LightningDataModule):
    """
    LightningDataModule wrapping PASTISRDataset.

    Standard PASTIS split: folds 1-4 = train, fold 5 = val/test.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        num_s2_frames: int = 10,
        num_s1_frames: int = 10,
        use_s1a: bool = True,
        use_s1d: bool = False,
        image_size: int = 128,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        s2_key: str = "S2",
        s1_key: str = "S1",
    ):
        """
        Args:
            data_root:          Path to PASTIS-R root directory.
            batch_size:         Samples per batch.
            num_workers:        DataLoader worker count.
            num_s2_frames:      Fixed temporal length for Sentinel-2.
            num_s1_frames:      Fixed temporal length for Sentinel-1.
            use_s1a:            Include ascending Sentinel-1 pass.
            use_s1d:            Include descending Sentinel-1 pass.
            image_size:         Spatial size (H=W) after optional resize.
                                Currently stored but not applied in __getitem__
                                (patches are already 128×128 in PASTIS-R).
            max_train_samples:  Cap training set size (reproducible subset).
            max_val_samples:    Cap validation set size.
            s2_key:             Dict key for Sentinel-2 tensors.
            s1_key:             Dict key for Sentinel-1 tensors.
        """
        super().__init__()
        self.data_root        = data_root
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.num_s2_frames    = num_s2_frames
        self.num_s1_frames    = num_s1_frames
        self.use_s1a          = use_s1a
        self.use_s1d          = use_s1d
        self.image_size       = image_size
        self.max_train_samples = max_train_samples
        self.max_val_samples  = max_val_samples
        self.s2_key           = s2_key
        self.s1_key           = s1_key

        self._train_ds: Optional[Union[PASTISRDataset, Subset]] = None
        self._val_ds:   Optional[Union[PASTISRDataset, Subset]] = None
        self._test_ds:  Optional[PASTISRDataset]                = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        common_kw = dict(
            root=self.data_root,
            num_s2_frames=self.num_s2_frames,
            num_s1_frames=self.num_s1_frames,
            use_s1a=self.use_s1a,
            use_s1d=self.use_s1d,
            s2_key=self.s2_key,
            s1_key=self.s1_key,
        )

        if stage in (None, "fit"):
            train_full = PASTISRDataset(
                folds=[1, 2, 3, 4],
                augment=True,
                max_samples=self.max_train_samples,
                **common_kw,
            )
            self._train_ds = train_full

            val_full = PASTISRDataset(
                folds=[5],
                augment=False,
                max_samples=self.max_val_samples,
                **common_kw,
            )
            self._val_ds = val_full

        if stage in (None, "validate"):
            if self._val_ds is None:
                self._val_ds = PASTISRDataset(
                    folds=[5],
                    augment=False,
                    max_samples=self.max_val_samples,
                    **common_kw,
                )

        if stage in (None, "test"):
            # Test uses the same fold 5 data without any sample cap
            self._test_ds = PASTISRDataset(
                folds=[5],
                augment=False,
                **common_kw,
            )

    # ------------------------------------------------------------------
    # Dataset properties  (terratorch base_task interface)
    # ------------------------------------------------------------------

    @property
    def train_dataset(self) -> Optional[PASTISRDataset]:
        ds = self._train_ds
        if ds is None:
            return None
        return ds.dataset if isinstance(ds, Subset) else ds

    @property
    def val_dataset(self) -> Optional[PASTISRDataset]:
        ds = self._val_ds
        if ds is None:
            return None
        return ds.dataset if isinstance(ds, Subset) else ds

    @property
    def test_dataset(self) -> Optional[PASTISRDataset]:
        return self._test_ds

    # ------------------------------------------------------------------
    # Collate function
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate a list of sample dicts into batched tensors.

        All temporal dimensions are fixed per dataset, so simple stacking works.
        """
        s2_key = list(batch[0]["image"].keys())[0]
        s1_key = list(batch[0]["image"].keys())[1]

        s2_list    = [s["image"][s2_key] for s in batch]
        s1_list    = [s["image"][s1_key] for s in batch]
        mask_list  = [s["mask"]          for s in batch]
        s2d_list   = [s["s2_dates"]      for s in batch]
        s1d_list   = [s["s1_dates"]      for s in batch]

        return {
            "image": {
                s2_key: torch.stack(s2_list,   dim=0),  # (B, T_s2, 10, H, W)
                s1_key: torch.stack(s1_list,   dim=0),  # (B, T_s1, C_s1, H, W)
            },
            "mask":     torch.stack(mask_list,  dim=0),  # (B, H, W)
            "s2_dates": torch.stack(s2d_list,   dim=0),  # (B, T_s2)
            "s1_dates": torch.stack(s1d_list,   dim=0),  # (B, T_s1)
        }

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def plot(self, sample, split: Optional[str] = None):
        """No-op plot method required by terratorch base_task."""
        return None
