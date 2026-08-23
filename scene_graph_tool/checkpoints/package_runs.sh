#!/usr/bin/env bash
set -e

# Helper script to package RLHP runs into zip archives for Google Drive upload
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNS_DIR="${ROOT_DIR}/scene_graph_tool/rlhp_runs"
OUTPUT_DIR="${ROOT_DIR}/scene_graph_tool/checkpoints/packaged_archives"

mkdir -p "${OUTPUT_DIR}"

echo "==================================================================="
echo " Packaging RLHP Runs for Google Drive Upload"
echo " Source: ${RUNS_DIR}"
echo " Output: ${OUTPUT_DIR}"
echo "==================================================================="

if [ ! -d "${RUNS_DIR}" ]; then
    echo "Error: ${RUNS_DIR} does not exist."
    exit 1
fi

cd "${RUNS_DIR}"

echo "1. Packaging Full RLHP Tuned Model..."
zip -r "${OUTPUT_DIR}/preference3_tuned_anchor_ret20_tau01.zip" preference3_tuned_anchor_ret20_tau01/

echo "2. Packaging Ablation 1 (Pure Preference)..."
zip -r "${OUTPUT_DIR}/preference3_pure_pref_setting_tau01.zip" preference3_pure_pref_setting_tau01/

echo "3. Packaging Ablation 2 (+Proto Anchor)..."
zip -r "${OUTPUT_DIR}/preference3_proto_anchor_only_tau01.zip" preference3_proto_anchor_only_tau01/

echo "4. Packaging Ablation 3 (+KL Retention)..."
zip -r "${OUTPUT_DIR}/preference3_retention_only_tau01.zip" preference3_retention_only_tau01/

echo "5. Packaging Audits & Baselines..."
zip -r "${OUTPUT_DIR}/audits_and_baselines.zip" preference3_audit_v6_loader/ preference3_v6_export_audit/ pure_pref_run1/ script.md

echo ""
echo "All archives created under ${OUTPUT_DIR}:"
ls -lh "${OUTPUT_DIR}"
