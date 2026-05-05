#!/bin/bash
set -e
cd /home/clara/terratorch-ibm
TERRATORCH=/home/clara/anaconda3/envs/terratorch311/bin/terratorch

echo "=========================================="
echo "[3/3] DOFA-Large multimodal (S2+S1 15-band)"
echo "=========================================="
$TERRATORCH fit --config examples/segmentation/dofa_dfc2020.yaml \
  > /tmp/train_dofa_dfc2020.log 2>&1
echo "DOFA done: $(date)"

echo "=========================================="
echo "[4/3] Prithvi-EO-300M (S2 only baseline)"
echo "=========================================="
$TERRATORCH fit --config examples/segmentation/prithvi_dfc2020.yaml \
  > /tmp/train_prithvi_dfc2020.log 2>&1
echo "Prithvi done: $(date)"

echo "All done: $(date)"
