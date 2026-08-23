#!/bin/bash
set -e

PROJECT_ROOT="/home/whopqa/Documents/Research/VLN/scenegraph/SGG-Benchmark"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-rlhp

echo "================================================================="
echo "[1/4] Training preference3_proto_anchor_only_tau01 (proto=20, ret=0, tau=0.1, lr=2e-6)"
echo "================================================================="
.venv/bin/python scene_graph_tool/rlhp.py train \
  --sessions scene_graph_tool/scene_graph_preferences3.csv \
  --image-root scene_graph_tool/train_00000_of_00016_images \
  --labels scene_graph_tool/psg_labels.json \
  --psg-test datasets/PSG/coco_format/test \
  --checkpoint checkpoint/best_model.pth \
  --config checkpoint/config.yml \
  --yolo-checkpoint checkpoint/PSG_yolo12m_backbone.pt \
  --device cuda \
  --runs-root scene_graph_tool/rlhp_runs \
  --run-id preference3_proto_anchor_only_tau01 \
  --tau 0.1 \
  --retention-weight 0.0 \
  --proto-anchor-weight 20.0 \
  --learning-rate 2e-6 \
  --weight-decay 0.01 \
  --gradient-accumulation 4 \
  --clip-norm 1.0 \
  --epochs 5 \
  --early-stopping-patience 5 \
  --save-every-epoch

echo "================================================================="
echo "[2/4] Evaluating preference3_proto_anchor_only_tau01 on PSG test"
echo "================================================================="
.venv/bin/python scene_graph_tool/rlhp.py evaluate \
  --run-dir scene_graph_tool/rlhp_runs/preference3_proto_anchor_only_tau01 \
  --baseline-checkpoint checkpoint/best_model.pth \
  --config checkpoint/config.yml \
  --data-dir datasets/PSG \
  --device cuda

echo "================================================================="
echo "[3/4] Training preference3_retention_only_tau01 (proto=0, ret=20, tau=0.1, lr=2e-6)"
echo "================================================================="
.venv/bin/python scene_graph_tool/rlhp.py train \
  --sessions scene_graph_tool/scene_graph_preferences3.csv \
  --image-root scene_graph_tool/train_00000_of_00016_images \
  --labels scene_graph_tool/psg_labels.json \
  --psg-test datasets/PSG/coco_format/test \
  --checkpoint checkpoint/best_model.pth \
  --config checkpoint/config.yml \
  --yolo-checkpoint checkpoint/PSG_yolo12m_backbone.pt \
  --device cuda \
  --runs-root scene_graph_tool/rlhp_runs \
  --run-id preference3_retention_only_tau01 \
  --tau 0.1 \
  --retention-weight 20.0 \
  --proto-anchor-weight 0.0 \
  --learning-rate 2e-6 \
  --weight-decay 0.01 \
  --gradient-accumulation 4 \
  --clip-norm 1.0 \
  --epochs 5 \
  --early-stopping-patience 5 \
  --save-every-epoch

echo "================================================================="
echo "[4/4] Evaluating preference3_retention_only_tau01 on PSG test"
echo "================================================================="
.venv/bin/python scene_graph_tool/rlhp.py evaluate \
  --run-dir scene_graph_tool/rlhp_runs/preference3_retention_only_tau01 \
  --baseline-checkpoint checkpoint/best_model.pth \
  --config checkpoint/config.yml \
  --data-dir datasets/PSG \
  --device cuda

echo "All trainings and evaluations completed successfully!"
