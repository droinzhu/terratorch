#!/bin/bash
# Wait for PASTIS-R download, unzip, then run ablation experiments:
#   ①  Prithvi S2-only, single frame  (baseline)
#   ②  BiCross-EO S2+S1, single frame (no temporal)
#   ③  BiCross-EO S2+S1 + LTAE T=10  (full model)

set -e

ZIP=/home/clara/trans/pastis_r/PASTIS-R.zip
DEST=/home/clara/trans/pastis_r
CD=/home/clara/terratorch-ibm
TERRATORCH=/home/clara/anaconda3/envs/terratorch311/bin/terratorch
DL_PID=$(pgrep -f "PASTIS-R.zip" | head -1)

echo "[$(date '+%H:%M:%S')] 等待下载完成 (PID: ${DL_PID:-未知})..."

while kill -0 "$DL_PID" 2>/dev/null; do
    DOWNLOADED=$(du -sh "$ZIP" 2>/dev/null | cut -f1)
    echo "[$(date '+%H:%M:%S')] 已下载: $DOWNLOADED / ~50GB"
    sleep 120
done

echo "[$(date '+%H:%M:%S')] 下载完成，验证文件大小..."
ACTUAL=$(du -sm "$ZIP" | cut -f1)
if [ "$ACTUAL" -lt 50000 ]; then
    echo "ERROR: 文件大小 ${ACTUAL}MB 异常，下载可能不完整"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] 开始解压 (~50GB，预计15-20分钟)..."
cd "$DEST"
unzip -o PASTIS-R.zip
echo "[$(date '+%H:%M:%S')] 解压完成"

if [ ! -d "$DEST/PASTIS-R/DATA_S2" ]; then
    echo "ERROR: 未找到 PASTIS-R/DATA_S2，请检查解压结果"
    exit 1
fi

cd "$CD"

# ── ① Prithvi S2-only single frame ──────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] === 实验① Prithvi S2-only (单帧 baseline) ==="
$TERRATORCH fit --config examples/segmentation/prithvi_pastis_r.yaml \
    > /tmp/train_prithvi_pastis_r.log 2>&1
echo "[$(date '+%H:%M:%S')] 实验① 完成"

# ── ② BiCross-EO single frame (no temporal) ─────────────────────────────────
echo "[$(date '+%H:%M:%S')] === 实验② BiCross-EO S2+S1 (单帧，无时序) ==="
$TERRATORCH fit --config examples/segmentation/bicross_eo_pastis_r_single.yaml \
    > /tmp/train_bicross_pastis_r_single.log 2>&1
echo "[$(date '+%H:%M:%S')] 实验② 完成"

# ── ③ BiCross-EO + LTAE T=10 ────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] === 实验③ BiCross-EO + LTAE (T=10 时序融合) ==="
$TERRATORCH fit --config examples/segmentation/bicross_eo_pastis_r.yaml \
    > /tmp/train_bicross_pastis_r.log 2>&1
echo "[$(date '+%H:%M:%S')] 实验③ 完成"

echo ""
echo "=============================="
echo "所有对比实验完成！"
echo "  ① /tmp/train_prithvi_pastis_r.log"
echo "  ② /tmp/train_bicross_pastis_r_single.log"
echo "  ③ /tmp/train_bicross_pastis_r.log"
echo "TensorBoard: tensorboard --logdir /home/clara/trans/tensorboard"
echo "=============================="
