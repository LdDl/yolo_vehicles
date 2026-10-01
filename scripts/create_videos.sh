#!/usr/bin/env bash
# Build a slideshow preview of public validation images, without temporal assumptions.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$(cd -- "${1:-$project_dir/datasets/vehicles/merged/images/val}" && pwd -P)"
output_file="$(realpath -m -- "${2:-$project_dir/videos/public-val.mp4}")"
if [[ -e "$output_file" ]]; then
  printf 'Output already exists: %s\n' "$output_file" >&2
  exit 1
fi
temporary_dir="$(mktemp -d -t vehicles-preview-XXXXXXXX)"
cleanup() {
  local file
  for file in "$temporary_dir/images.list" "$temporary_dir/frames.txt"; do
    if [[ -f "$file" ]]; then
      rm -- "$file"
    fi
  done
  rmdir -- "$temporary_dir"
}
trap cleanup EXIT

find -L "$source_dir" -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0 \
  | LC_ALL=C sort -z > "$temporary_dir/images.list"
images=()
mapfile -d '' -t -n 100 images < "$temporary_dir/images.list"
if (( ${#images[@]} == 0 )); then
  printf 'No images in %s\n' "$source_dir" >&2
  exit 1
fi

listing="$temporary_dir/frames.txt"
# Repeat the last entry so FFmpeg can apply its duration.
for image in "${images[@]}" "${images[-1]}"; do
  if [[ "$image" == *$'\n'* || "$image" == *$'\r'* ]]; then
    printf 'Unsupported filename: %q\n' "$image" >&2
    exit 1
  fi
  escaped="${image//\'/\'\\\'\'}"
  printf "file '%s'\nduration 1\n" "$escaped"
done > "$listing"

mkdir -p -- "$(dirname -- "$output_file")"
ffmpeg -n -f concat -safe 0 -i "$listing" \
  -vf 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1' \
  -r 1 -frames:v "${#images[@]}" -c:v libx264 \
  -pix_fmt yuv420p "$output_file"
