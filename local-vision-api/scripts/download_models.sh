#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-"$ROOT_DIR/models"}"

mkdir -p "$MODEL_DIR"

download() {
  local name="$1"
  local url="$2"
  local out="$MODEL_DIR/$name"

  if [[ -f "$out" ]]; then
    echo "Exists: $out"
    return 0
  fi

  echo "Downloading: $name"
  curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
}

download "yolov8x-worldv2.pt" "https://github.com/ultralytics/assets/releases/latest/download/yolov8x-worldv2.pt"
download "sam2_l.pt" "https://github.com/ultralytics/assets/releases/latest/download/sam2_l.pt"
download "yolo11x-pose.pt" "https://github.com/ultralytics/assets/releases/latest/download/yolo11x-pose.pt"
download "yolo11x-cls.pt" "https://github.com/ultralytics/assets/releases/latest/download/yolo11x-cls.pt"

echo "Done. Models in: $MODEL_DIR"

