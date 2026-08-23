#!/usr/bin/env bash
set -e

# Google Drive Checkpoint Downloader for Scene Graph Preference Alignment
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1CRyhpJNsR1Ht4Kay5SEpRqtrl-uTk-Bw?usp=sharing"
GDRIVE_FOLDER_ID="1CRyhpJNsR1Ht4Kay5SEpRqtrl-uTk-Bw"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${ROOT_DIR}/scene_graph_tool/rlhp_runs"

echo "==================================================================="
echo " Downloading RLHP Pretrained Checkpoints & Runs from Google Drive"
echo " Target Directory: ${TARGET_DIR}"
echo " Folder URL: ${GDRIVE_FOLDER_URL}"
echo "==================================================================="

mkdir -p "${TARGET_DIR}"

# Check if gdown is available
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown for Google Drive batch download..."
    pip install --upgrade gdown
fi

echo "Fetching folders from Google Drive..."
gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" -O "${TARGET_DIR}" --remaining-ok || {
    echo ""
    echo "Note: If automated download encounters rate limits, please manually download"
    echo "the folders from: ${GDRIVE_FOLDER_URL}"
    echo "and place them inside '${TARGET_DIR}'."
}

echo ""
echo "Download complete! Checkpoints are available under: ${TARGET_DIR}"
