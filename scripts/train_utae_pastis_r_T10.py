"""
Fair comparison: UTAE (Garnot, ICCV 2021) on PASTIS-R with T=10
Same data as BiCross-EO T=10 run for direct comparison.
"""
import sys
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

sys.path.insert(0, "/tmp/utae-paps/src")
sys.path.insert(0, "/home/clara/terratorch-ibm")

from backbones.utae import UTAE
from terratorch.datamodules import PASTISRDataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_ROOT    = "/home/clara/trans/pastis_r/PASTIS-R"
NUM_CLASSES  = 20
IGNORE_INDEX = 19
T            = 10
BATCH_SIZE   = 4
NUM_WORKERS  = 4
MAX_TRAIN    = 5000
MAX_VAL      = 500
MAX_EPOCHS   = 100
PATIENCE_ES  = 20
PATIENCE_LR  = 6
LR_FACTOR    = 0.5
LR           = 6e-4
WEIGHT_DECAY = 0.05
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR      = "/home/clara/trans/tensorboard/utae_pastis_r_T10"
CKPT_DIR     = "/home/clara/trans/checkpoint/utae_pastis_r_T10"


def build_model() -> nn.Module:
    return UTAE(
        input_dim=10,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, NUM_CLASSES],
        n_head=16,
        d_model=256,
        d_k=4,
        encoder_norm="group",
        agg_mode="att_group",
    )


def build_metrics(prefix: str) -> MetricCollection:
    return MetricCollection({
        "mIoU": MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, average="macro"),
        "Pixel_Accuracy": MulticlassAccuracy(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, average="micro"),
    }, prefix=prefix)


def run_epoch(model, loader, criterion, optimizer, metrics, device, train: bool):
    model.train() if train else model.eval()
    metrics.reset()
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            img = batch["image"]
            s2 = img["S2"].to(device) if isinstance(img, dict) else img.to(device)
            mask = batch["mask"].to(device)
            dates = batch.get("s2_dates", None)
            if dates is not None:
                dates = dates.to(device)

            logits = model(s2, batch_positions=dates)
            loss = criterion(logits, mask)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            metrics.update(logits.argmax(dim=1), mask)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), metrics.compute()


def main():
    dm = PASTISRDataModule(
        data_root=DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        num_s2_frames=T, num_s1_frames=T, use_s1a=True, use_s1d=False,
        image_size=128, max_train_samples=MAX_TRAIN, max_val_samples=MAX_VAL,
        single_frame=False, single_modality=None,
    )
    dm.setup("fit")
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()
    log.info(f"Train: {len(dm._train_ds)} | Val: {len(dm._val_ds)}")

    model = build_model().to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"UTAE T={T}  trainable={n_train/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE_LR, factor=LR_FACTOR)

    train_metrics = build_metrics("train/").to(DEVICE)
    val_metrics   = build_metrics("val/").to(DEVICE)

    from torch.utils.tensorboard import SummaryWriter
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)

    best_miou, best_val_loss, no_improve = 0.0, float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_res = run_epoch(model, train_loader, criterion, optimizer, train_metrics, DEVICE, True)
        val_loss, val_res     = run_epoch(model, val_loader,   criterion, optimizer, val_metrics,   DEVICE, False)

        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_mIoU={train_res['train/mIoU']:.4f} | "
            f"val_loss={val_loss:.4f} val_mIoU={val_res['val/mIoU']:.4f} | lr={cur_lr:.1e}"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss",   val_loss,   epoch)
        writer.add_scalar("val/mIoU",   val_res["val/mIoU"].item(), epoch)
        writer.add_scalar("train/mIoU", train_res["train/mIoU"].item(), epoch)
        writer.add_scalar("lr",         cur_lr,     epoch)

        if val_res["val/mIoU"].item() > best_miou:
            best_miou = val_res["val/mIoU"].item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE_ES:
                log.info(f"EarlyStopping at epoch {epoch}. Best val/mIoU={best_miou:.4f}")
                break

    writer.close()
    log.info(f"=== UTAE T={T} finished. Best val/mIoU = {best_miou:.4f} ===")


if __name__ == "__main__":
    main()
