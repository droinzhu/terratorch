# What happens when you run `terratorch fit -c <config.yaml>` (AED object detection)

This guide explains (in easy words) why you see **tiling** and **caching** when you train the AED (African Elephant Dataset) object detection examples.

## Backbone (the simple 8-step version)
This is the original “backbone” checklist for the page (kept as the base):

1- you run the command
2- config file is read
3- classes are created (Datamodule, Task, Trainer)
4- dataset (tiling) happens
5- dataloader batches are formed
6- model is built (framework + backbone)
7- training loop starts (ft)
8- outputs are checkpoints, logs, and tile cache

## Big idea (one sentence)
AED images are **very large**, so TerraTorch splits each big image into many smaller **tiles** (like 512×512 crops) that the model can train on, and it **caches** those tiles on disk so it doesn’t have to re-cut them every run.

---

## Step-by-step (expanded)

### 1) You run the command
Example:

`terratorch fit -c examples/confs/object_detection_aed_elephants.yaml`

Meaning: “start training, and use this config file to decide everything (data, model, trainer settings).”

### 2) TerraTorch reads the config
The YAML config contains:
- **trainer** settings (epochs, devices, logging, checkpointing)
- **data**: which `DataModule` class to create + its arguments
- **model**: which `Task` class to create + model factory arguments

### 3) TerraTorch creates 3 main objects
At runtime it creates:
- **DataModule**: responsible for making datasets + dataloaders
- **Task**: the LightningModule (training/validation/test logic, metrics)
- **Trainer**: the engine that runs the training loop

### 4) Dataset tiling happens (this is the important part)

#### What “tiling” means
AED images can be *huge* (thousands of pixels wide/high). Training Faster R-CNN / Mask R-CNN directly on huge images is slow and often runs out of GPU memory.

So TerraTorch does this:
- take one big image
- cut it into many smaller crops called **tiles** (example: 512×512)
- each tile becomes one training sample

#### Why there is “overlap”
Tiles can overlap (e.g. overlap=128). This helps because an elephant near the border of one tile might be fully visible in a neighboring tile.

#### What happens to bounding boxes
Bounding boxes are stored in pixel coordinates. When you cut a tile from a big image:
- boxes are **shifted** into the tile coordinate system (subtract the tile’s top-left x/y)
- boxes are **clipped** to stay inside the tile
- boxes with too little overlap with the tile can be dropped

#### Tiling cache is built (why you see `tile_cache_*` folders)

#### What “tile cache” means
Cutting tiles and fixing boxes is expensive. TerraTorch writes the tiles to disk so future runs are much faster.

You typically see folders like:
- `tile_cache_train/`
- `tile_cache_test/`

Inside, each tile is stored as:
- a PNG image: `tile_<imageIndex>_<x0>_<y0>.png`
- a JSON file with boxes/labels: `tile_<imageIndex>_<x0>_<y0>.json`

#### What happens on the first run vs later runs
- **First run**: “build tiles” → slow (it creates PNG/JSON files)
- **Later runs**: “reuse cached tiles” → fast (it just loads PNG/JSON)

### 5) Dataloader batches are formed

Object detection batches can’t be a single tensor for boxes, because each image has a different number of boxes.

So the DataLoader usually returns a batch shaped like:
- `image`: a tensor `[B, C, H, W]` (tiles stacked)
- `boxes`: a list of length `B`, each item is a tensor `[N_i, 4]`
- `labels`: a list of length `B`, each item is a tensor `[N_i]`

Where `N_i` can be different per tile.

### 6) The model is built (framework + backbone)
The model factory builds two main things:
- **framework**: e.g. Faster R-CNN or Mask R-CNN (the detection “head” + loss logic)
- **backbone**: the feature extractor (e.g. TerraMind/Prithvi/timm backbone)

The backbone turns each tile into feature maps, and the framework predicts boxes/labels (and sometimes masks) from those features.

### 7) Training loop starts (fine-tuning)
The Trainer runs:
- forward pass on a batch of tiles
- compute detection losses
- backward + optimizer step
- periodic validation (mAP metrics)

### 8) Outputs (checkpoints, logs, tile cache)
You typically get:
- **checkpoints** (model weights)
- **logs** (TensorBoard/W&B/etc)
- **tile cache** folders (so you don’t rebuild tiles every time)

---

## What TerraTorch expects your AED data to look like

### Before tiling (base dataset)
TerraTorch expects COCO-style data:
- an **image folder** (JPEG/PNG/etc)
- a COCO **annotation JSON** containing bounding boxes

The base dataset returns a dict like:
- `image`: torch tensor `[C, H, W]`
- `boxes`: torch tensor `[N, 4]` in **XYXY pixel coordinates**
- `labels`: torch tensor `[N]`

### After tiling (tiled dataset)
You get the same keys, but:
- `image` is now a **tile** (e.g. `[3, 512, 512]`)
- `boxes/labels` only contain objects that overlap the tile, and the box coordinates are relative to the tile.

---

## Common “gotchas” (quick)
- If you delete the `tile_cache_*` folders, the next run will rebuild tiles (slow again).
- If you change `tile_size` or `overlap`, old cached tiles no longer match; rebuild the cache.
- Some configs skip tiles that have no boxes (to avoid training mostly on empty background).
