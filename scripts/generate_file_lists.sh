#!/usr/bin/env bash
# Regenerate local paths and adjacent Darknet labels for the merged public dataset.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$project_dir/scripts/prepare_dataset.py" configure \
  --dataset "${1:-$project_dir/datasets/vehicles/merged}" \
  --output "$project_dir/data" --backup "$project_dir/weights" --darknet-labels
