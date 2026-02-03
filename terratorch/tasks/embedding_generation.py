from pathlib import Path
import warnings
import logging
import json
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq

from terratorch.tasks.base_task import TerraTorchTask
from terratorch.registry import MODEL_FACTORY_REGISTRY

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.simplefilter("once", UserWarning)
logger = logging.getLogger("EmbeddingGenerationTask")

class EmbeddingGenerationTask(TerraTorchTask):
    """
    Task that runs inference over model backbone to generate and save embeddings.
    """

    def __init__(
            self,
            model_args: dict,
            output_dir: str = "embeddings",
            embed_file_key: str = "filename",
            layers: list[int] = [-1],
            output_format: str = "tiff",
            has_cls: bool = False,
            embedding_pooling: str | None = None,
            freeze_backbone = True
    ) -> None:
        """Constructor for EmbeddingGenerationTask

        Args:
            model (str): Model name from backbone registry.
            model_args (dict, optional): Arguments passed to the model factory. Defaults to None.
            output_dir (str, optional): Directory to save embeddings. Defaults to "embeddings".
            embed_file_key (str, optional): Identifier key for single file ids in input data, will be used as embedding identifiers. Defaults to "filename".
            layers (list[int], optional): List of layers to extract embeddings from. Defaults to [-1].
            output_format (str, optional): Format for saving embeddings ('tiff' for GeoTIFF, 'parquet' for GeoParquet). Defaults to "tiff".
            has_cls (bool): Whether the model has a CLS token. Defaults to False.
            embedding_pooling (str | None, optional): Pooling method for embeddings. Defaults to None.
        """
        self.output_format = output_format.lower()
        self._parquet_writer = None
        self._parquet_path = None

        if self.output_format not in ("tiff", "parquet", "parquet_joint"):
            raise ValueError(
                f"Unsupported output format: {self.output_format}. "
                "Supported formats are 'tiff', 'parquet', 'parquet_joint'."
            )

        self._config_saved = False
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.embed_file_key = embed_file_key
        self.has_cls = has_cls
        self.embedding_pooling = embedding_pooling
        self.embedding_indices = layers

        model_args = model_args or {}
        if model_args.get("necks", None):
            logger.info(
                "EmbeddingGeneration is designed to automatically add necks based on the selected "
                "output format and aggregation settings. Since necks were provided explicitly, "
                "automatic neck insertion and embedding aggregation are skipped. "
                "This may cause incompatibilities with the chosen output format."
            )
        else:
            if embedding_pooling in (None, "None", "keep"):
                model_args["necks"] = [
                    {
                        "name": "SelectIndices",
                        "indices": self.embedding_indices
                    }
                ]
                if output_format == "tiff":
                    model_args["necks"].append(
                        {
                            "name": "ReshapeTokensToImage",
                            "remove_cls_token": self.has_cls
                        }
                    )
                    logger.info(
                        "GeoTIFF selected; 2D token embeddings (ViT) will be reshaped to "
                        "[C, sqrt(num_tokens), sqrt(num_tokens)] after dropping CLS if present."
                    )
            elif embedding_pooling in ["mean", "max", "min", "cls"]:
                model_args["necks"] = [
                    {
                        "name": "AggregateTokens",
                        "pooling": embedding_pooling,
                        "indices": self.embedding_indices,
                        "drop_cls": has_cls
                    }
                ]
                if self.output_format == "tiff":
                    warnings.warn("GeoTIFF output not recommended with embedding pooling, saves 1D vectors as (C,1,1).")
            else:
                raise ValueError(f"EmbeddingPooling {embedding_pooling} is not supported.")

        self.model_args = model_args
        self.aux_heads = []
        self.model_factory = MODEL_FACTORY_REGISTRY.build("EncoderDecoderFactory")
        super().__init__(task="embedding_generation")

    def infer_BT(self, x: torch.Tensor | dict[str, torch.Tensor]) -> tuple[int, int]:
        """Infer (B, T). For 5D assume [B, C, T, H, W] as standardized by TemporalWrapper."""
        if isinstance(x, dict):
            v = next(iter(x.values()))
        else:   
            v = x
        B = v.shape[0]
        T = v.shape[2] if v.ndim == 5 else 1 
        return B, T  

    def check_file_ids(
        self,
        file_ids: torch.Tensor | np.ndarray | list | tuple,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> None:
        """Validate `file_ids` matches (B,) or (B, T) inferred from `x`."""
        B, T = self.infer_BT(x)

        if isinstance(file_ids, (torch.Tensor, np.ndarray)):
            expected = (B,) if T == 1 else (B, T)
            if tuple(file_ids.shape) != expected:
                raise ValueError(f"`file_ids` shape mismatch: expected {expected}, got {tuple(file_ids.shape)}")
            return

        if isinstance(file_ids, (list, tuple)):
            if len(file_ids) != B:
                raise ValueError(f"`file_ids` length mismatch: expected {B}, got {len(file_ids)}")
            if T > 1 and isinstance(file_ids[0], (list, tuple, np.ndarray)) and len(file_ids[0]) != T:
                raise ValueError(f"`file_ids` must have inner length {T}, got {len(file_ids[0])}")
            return

        raise TypeError("`file_ids` must be a tensor/ndarray or a (nested) list/tuple")


    def save_configuration_summary(
            self,
            x: torch.Tensor | dict[str, torch.Tensor],
    ) -> None:
        """
        Saves a JSON containing model, layer configuration, and output specs.
        """
        if self._config_saved:
            return

        outputs = self.model.encoder(x)

        if not isinstance(outputs, list):
            outputs = [outputs]
        n_outputs = len(outputs)

        resolved_indices = [
            (idx if idx >= 0 else n_outputs + idx) for idx in self.embedding_indices
        ]

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6

        config_summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(self.output_path.absolute()),
            "output_format": self.output_format,
            "backbone": self.model_args["backbone"] ,
            "backbone_total_params_million": total_params,
            "has_cls": self.has_cls,
            "embedding_pooling": self.embedding_pooling,
            "model_layer_count": n_outputs,
            "n_layers_saved": len(self.embedding_indices),
            "layers": [
                {   "output_folder_name": f"layer_{i:02d}",
                    "requested_index": folder,
                    "layer_number": res + 1,
                    "layer_output_shape": list(outputs[res][0].shape)
                }
                for i, (folder, res) in enumerate(
                    zip(self.embedding_indices, resolved_indices)
                )
            ],
        }

        out_path = self.output_path / "configuration_summary.json"
        try:
            with open(out_path, "w") as f:
                json.dump(config_summary, f, indent=2)
            logger.info(f"Configuration summary saved to {out_path}")
        except IOError as e:
            logger.error(f"Failed to write configuration summary: {e}")

        self._config_saved = True


    @torch.no_grad()
    def predict_step(self, batch: dict) -> None:
        embed_file_key = self.embed_file_key
        x = batch['image']

        if isinstance(x, dict) and embed_file_key in x:
            file_ids = x.pop(embed_file_key)
            metadata = self.pull_metadata(x)
        else:
            file_ids = batch.get(embed_file_key)
            if file_ids is None:
                raise KeyError(f"Key '{embed_file_key}' not found in input dictionary.")
            if 'metadata' in batch:
                metadata = self.pull_metadata(batch['metadata'])
            else:   
                metadata = self.pull_metadata(batch)

        self.check_file_ids(file_ids, x)
        embeddings = self(x)
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        self.save_configuration_summary(x)
        for layer, embeddings_per_layer in enumerate(embeddings):
            self.save_embeddings(embeddings_per_layer, file_ids, metadata, layer)

    def on_predict_end(self) -> None:
        writer = getattr(self, "_parquet_writer", None)
        if writer is not None:
            writer.close()
            self._parquet_writer = None

    def save_embeddings(
        self,
        embedding: torch.Tensor | dict[str, torch.Tensor],
        file_ids: list[str] | None,
        metadata: dict,
        layer: int,
    ) -> None:
        """Save embeddings for a given layer (per sample, optional per timestep and per modality)."""
        path = self.output_path / f"layer_{layer:02d}"
        if isinstance(embedding, dict):
            for modality, t in embedding.items():
                path = path / modality
                self.write_batch(t, file_ids, metadata, path)
        elif isinstance(embedding, torch.Tensor):
            self.write_batch(embedding, file_ids, metadata, path)
        else:
            raise TypeError(f"Unsupported embedding type: {type(embedding)}. Expected Tensor or dict of Tensors.")

    def pull_metadata(
        self, 
        data: dict
    ) -> dict:
        """Extract known metadata fields from `batch`, removing them from data and returning a metadata dict.
        Args:
            data (dict): Input data dictionary containing metadata.
        Returns:
            dict: Metadata dictionary.
        """
        def pop_first(d: dict, keys):
            for k in keys:
                if k in d:
                    return d.pop(k)
            return None
        
        # Aliases in priority order
        metadata_map = {
            "file_id":       ("file_id",),
            "product_id":    ("product_id",),
            "time":          ("time", "time_", "timestamp"),
            "grid_cell":     ("grid_cell",),
            "grid_row_u":    ("grid_row_u",),
            "grid_col_r":    ("grid_col_r",),
            "geometry":      ("geometry",),
            "utm_footprint": ("utm_footprint",),
            "crs":           ("crs", "utm_crs"),
            "pixel_bbox":    ("pixel_bbox",),
            "bounds":        ("bounds",),
            "center_lat":    ("center_lat", "centre_lat"),
            "center_lon":    ("center_lon", "centre_lon"),
        } 

        metadata = {}

        for key, aliases in metadata_map.items():
            value = pop_first(data, aliases)
            if value is not None:
                metadata[key] = value
        
        return metadata

    def write_batch(
            self,
            embedding: torch.Tensor,
            file_ids: list[str],
            metadata: dict,
            dir_path: Path,
    ) -> None:  
        """" Write a batch (optionally with timesteps) to GeoTIFF/GeoParquet."""
        dir_path.mkdir(parents=True, exist_ok=True)

        if not file_ids:
            return

        is_temporal = isinstance(file_ids[0], (list, tuple, np.ndarray))
        emb_np = embedding.detach().cpu().numpy()

        if self.output_format == "parquet_joint":
            self.write_parquet_batch(emb_np, file_ids, metadata, is_temporal, dir_path)
            return

        tasks = list(self.iter_samples(emb_np, file_ids, metadata, is_temporal))
        if self.output_format == "tiff":
            writer = self.write_tiff
        elif self.output_format == "parquet":
            writer = self.write_parquet
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format!r}")

        max_workers = min(len(tasks), getattr(self, "num_workers", 16))

        def write_one(task):
            arr, filename, meta = task
            writer(arr, filename, meta, dir_path)

        if max_workers <= 1 or len(tasks) <= 1:
            for task in tasks:
                write_one(task)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(write_one, tasks))

    def iter_samples(
        self,
        embedding: torch.Tensor,
        file_ids: list[str],
        metadata: dict,
        is_temporal: bool,
    ) -> iter:
        """Yields (embedding_np, filename, metadata_sample) tuples."""
        B = len(file_ids)

        if is_temporal:
            for b in range(B):
                T = len(file_ids[b])
                for t in range(T):
                    filename = file_ids[b][t]
                    meta = {k: v[b][t] for k, v in metadata.items()}
                    arr = embedding[b, t, ...]
                    yield arr, filename, meta
        else:
            for b in range(B):
                filename = file_ids[b]
                meta = {k: v[b] for k, v in metadata.items()}
                arr = embedding[b, ...]
                yield arr, filename, meta

    def write_tiff(
        self,
        arr: np.ndarray,
        filename: str,
        metadata: dict,
        dir_path: Path
        ) -> None:
        """Write a single sample to GeoTIFF."""
        filename = Path(filename).stem
        out_path = dir_path / f"{filename}_embedding.tif"

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1, 1)

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=arr.shape[1],
            width=arr.shape[2],
            count=arr.shape[0],
            dtype=arr.dtype
        ) as dst:
            dst.write(arr)
            dst.update_tags(**{k: str(v) for k, v in metadata.items()})

    def write_parquet(
        self,
        arr: np.ndarray,
        filename: str,
        metadata: dict,
        dir_path: Path
    ) -> None:
        """Write a single sample to GeoParquet."""
        filename = Path(filename).stem
        out_path = dir_path / f"{filename}_embedding.parquet"
        row = {"embedding": arr.tolist()}
        row.update({k: (v.tolist() if v.ndim else v.item()) for k, v in metadata.items()})

        df = gpd.GeoDataFrame([row])
        df.to_parquet(out_path, index=False)

    def close_parquet(self) -> None:
        if getattr(self, "_parquet_writer", None) is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
            self._parquet_path = None

    def write_parquet_batch(
            self,
            emb_np: np.ndarray,
            file_ids: list[str],
            metadata: dict,
            is_temporal: bool,
            dir_path: Path,
        ) -> None:

            if is_temporal: # In the case of temporal data, we use 'iter_samples' logic to flatten the temporal and batch dimension
                tasks = list(self.iter_samples(emb_np, file_ids, metadata, is_temporal))
                filenames = []
                emb_list = []
                meta_cols: dict = {k: [] for k in metadata.keys()}

                for arr, filename, meta in tasks:
                    filenames.append(Path(filename).stem)
                    emb_list.append(np.asarray(arr, dtype=np.float32).reshape(-1))
                    for k, v in meta.items():
                        meta_cols[k].append(v)

                dim = emb_list[0].shape[0]
                flat = np.concatenate(emb_list, axis=0)
                emb_array = pa.FixedSizeListArray.from_arrays(pa.array(flat), dim)

            else: # In the non-temporal case we can directly treat batch dimension as row dimension
                filenames = file_ids
                meta_cols = metadata
                dim = emb_np.shape[1]
                emb_array = pa.FixedSizeListArray.from_arrays(emb_np.flatten(), dim)

            arrays = {
                "file_id": pa.array(filenames),
                "embedding": emb_array,
            }

            for k, vals in meta_cols.items():
                norm_vals = []
                for v in vals:
                    if hasattr(v, "item"):
                        norm_vals.append(v.item())
                    else:
                        norm_vals.append(v)
                arrays[k] = pa.array(norm_vals)

            table = pa.table(arrays)

            out_path = dir_path / "embeddings.parquet"
            if self._parquet_writer is None:
                self._parquet_path = out_path
                self._parquet_writer = pq.ParquetWriter(
                    out_path,
                    table.schema,
                    compression="snappy",
                )

            self._parquet_writer.write_table(table, row_group_size=len(filenames))