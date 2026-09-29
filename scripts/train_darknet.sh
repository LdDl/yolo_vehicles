#!/usr/bin/env bash
# Train a fresh public-dataset run or resume its existing checkpoint.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
model="${1:-}"
case "$model" in
  v3-tiny) base=yolov3-tiny; cutoff=11 ;;
  v4-tiny) base=yolov4-tiny; cutoff=29 ;;
  *) printf 'Usage: %s v3-tiny|v4-tiny [--resume checkpoint.weights | --scratch]\n' "$0"; exit 1 ;;
esac
shift
mode=pretrained
checkpoint=""
if [[ ${1:-} == --resume && $# == 2 ]]; then
  mode=resume
  checkpoint="$2"
elif [[ ${1:-} == --scratch && $# == 1 ]]; then
  mode=scratch
elif (( $# )); then
  printf '%s\n' 'Expected --resume checkpoint.weights or --scratch' >&2
  exit 1
fi
command -v darknet >/dev/null
cfg="$project_dir/configs/$base-vehicles.cfg"
source_data="$project_dir/data/generated/vehicles.data"
run="$project_dir/weights/$base-vehicles"
data="$project_dir/data/generated/$base-vehicles.data"
[[ -s "$source_data" ]] || { printf '%s\n' 'Run prepare_dataset.py configure --darknet-labels first.' >&2; exit 1; }
if [[ $mode != resume && -d "$run" ]]; then
  printf 'Run already exists: %s; use --resume.\n' "$run" >&2
  exit 1
fi
initial=()
flags=(-map -dont_show -mAP_epochs 1)
if [[ $mode == resume ]]; then
  [[ -s "$checkpoint" ]]
  previous_best="$run/$base-vehicles_best.weights"
  if [[ -s "$previous_best" ]]; then
    saved_best="$(mktemp "$run/$base-vehicles_best-before-resume.XXXXXX.weights")"
    cp "$previous_best" "$saved_best"
    printf 'Previous best preserved: %s\n' "$saved_best"
  fi
  initial=("$checkpoint")
elif [[ $mode == pretrained ]]; then
  pretrained="$project_dir/weights/pretrained"
  [[ -s "$pretrained/$base.weights" && -s "$pretrained/$base-coco.cfg" ]] || {
    printf 'Run: python3 scripts/download_pretrained.py --model %s\n' "$model" >&2
    exit 1
  }
  partial="$pretrained/$base.conv.$cutoff"
  if [[ ! -s "$partial" ]]; then
    darknet partial "$pretrained/$base-coco.cfg" "$pretrained/$base.weights" "$partial.part" "$cutoff"
    [[ -s "$partial.part" ]]
    mv "$partial.part" "$partial"
  fi
  initial=("$partial")
  flags+=(-clear)
else
  flags+=(-clear)
fi
mkdir -p "$run" "$(dirname "$data")"
awk -v backup="$run" '
  /^[[:space:]]*backup[[:space:]]*=/ { print "backup = " backup; next }
  { print }
' "$source_data" > "$data"
cp "$cfg" "$run/$base-vehicles.cfg"
cp "$project_dir/configs/$base-vehicles-infer.cfg" "$run/$base-vehicles-infer.cfg"
darknet detector train "$data" "$cfg" "${initial[@]}" "${flags[@]}" 2>&1 | tee -a "$run/train.log"
