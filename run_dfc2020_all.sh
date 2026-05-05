#!/bin/bash
# Train all 4 models on full DFC2020 dataset (35k train, 500 val cap)
set -e
cd /home/clara/terratorch-ibm
TERRATORCH=/home/clara/anaconda3/envs/terratorch311/bin/terratorch

echo "====== BiCross-EO (S2+S1 multimodal) ======"
$TERRATORCH fit --config examples/segmentation/bicross_eo_dfc2020.yaml \
  > /tmp/train_bicross_dfc2020_full.log 2>&1
echo "BiCross-EO done: $(date)"

echo "====== TerraMind (S2+S1 multimodal) ======"
$TERRATORCH fit --config examples/segmentation/terramind_dfc2020.yaml \
  > /tmp/train_terramind_dfc2020_full.log 2>&1
echo "TerraMind done: $(date)"

echo "====== DOFA (S2 optical) ======"
$TERRATORCH fit --config examples/segmentation/dofa_dfc2020.yaml \
  > /tmp/train_dofa_dfc2020_full.log 2>&1
echo "DOFA done: $(date)"

echo "====== Prithvi (S2 optical) ======"
$TERRATORCH fit --config examples/segmentation/prithvi_dfc2020.yaml \
  > /tmp/train_prithvi_dfc2020_full.log 2>&1
echo "Prithvi done: $(date)"

echo "===== All training complete: $(date) ====="
