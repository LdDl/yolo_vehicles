#!/usr/bin/env bash
# Train a fresh public-dataset run or resume its existing checkpoint.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
model="${1:-}"
case "$model" in
  v3-tiny) base=yolov3-tiny; cutoff=11 ;;
  v4-tiny) base=yolov4-tiny; cutoff=29 ;;
  *) printf 'Usage: %s v3-tiny|v4-tiny [--task vehicles|plates] [--resume checkpoint.weights | --scratch]\n' "$0"; exit 1 ;;
esac
shift
mode=pretrained
checkpoint=""
task=vehicles
while (( $# )); do
  case "$1" in
    --task)
      [[ $# -ge 2 && ( $2 == vehicles || $2 == plates ) ]] || { echo 'Expected --task vehicles|plates' >&2; exit 1; }
      task="$2"
      shift 2
      ;;
    --resume)
      [[ $# -ge 2 && $mode == pretrained ]] || { echo 'Expected one initialization mode and a checkpoint' >&2; exit 1; }
      mode=resume
      checkpoint="$2"
      shift 2
      ;;
    --scratch)
      [[ $mode == pretrained ]] || { echo 'Initialization modes are mutually exclusive' >&2; exit 1; }
      mode=scratch
      shift
      ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
  esac
done
config_dir="$project_dir/configs"
if [[ $task == plates ]]; then
  config_dir="$project_dir/plates/configs"
fi
command -v darknet >/dev/null
cfg="$config_dir/${base}-${task}.cfg"
source_data="$project_dir/data/generated/$task.data"
run="$project_dir/weights/${base}-${task}"
data="$project_dir/data/generated/${base}-${task}.data"
[[ -s "$source_data" ]] || { printf '%s\n' 'Generate the dataset training files first; see the task README.' >&2; exit 1; }
if [[ $mode != resume && -d "$run" ]]; then
  printf 'Run already exists: %s; use --resume.\n' "$run" >&2
  exit 1
fi
initial=()
flags=(-map -dont_show -mAP_epochs 1)
if [[ $mode == resume ]]; then
  [[ -s "$checkpoint" ]]
  previous_best="$run/${base}-${task}_best.weights"
  if [[ -s "$previous_best" ]]; then
    saved_best="$(mktemp "$run/${base}-${task}_best-before-resume.XXXXXX.weights")"
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
cp "$cfg" "$run/${base}-${task}.cfg"
cp "$config_dir/${base}-${task}-infer.cfg" "$run/${base}-${task}-infer.cfg"
darknet detector train "$data" "$cfg" "${initial[@]}" "${flags[@]}" 2>&1 | tee -a "$run/train.log"
