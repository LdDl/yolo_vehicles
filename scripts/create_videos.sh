#!/bin/bash
# Create videos from dataset images per source (camera/vid)

DATASET_DIR="/home/dimitrii/python_work/vehicles_yolo/aic_hcmc2020/images"
OUTPUT_DIR="/home/dimitrii/python_work/vehicles_yolo/videos"
FPS=25

create_video() {
    local split=$1  # train or val
    local source=$2 # cam_01, cam_02, ..., vid1, vid2, ...
    local input_dir="$DATASET_DIR/$split"
    local output_file="$OUTPUT_DIR/$split/${source}.mp4"
    local tmp_list="/tmp/ffmpeg_list_${split}_${source}.txt"

    # Find all JPGs for this source and create file list
    find "$input_dir" -name "${source}_*.jpg" | sort > "$tmp_list"

    count=$(wc -l < "$tmp_list")
    if [ "$count" -eq 0 ]; then
        echo "No images found for $split/$source"
        rm -f "$tmp_list"
        return
    fi

    echo "Creating $output_file from $count frames..."

    # Convert file list to ffmpeg concat format
    local concat_list="/tmp/ffmpeg_concat_${split}_${source}.txt"
    while read -r f; do
        echo "file '$f'"
        echo "duration 0.04"  # 1/25 fps
    done < "$tmp_list" > "$concat_list"

    # Create video using concat demuxer
    ffmpeg -y -f concat -safe 0 -i "$concat_list" \
        -c:v libx264 -preset fast -crf 23 \
        -pix_fmt yuv420p \
        "$output_file" 2>/dev/null

    rm -f "$tmp_list" "$concat_list"
    echo "Done: $output_file"
}

# Process train
echo "=== Processing TRAIN ==="
for i in $(seq -w 1 25); do
    create_video train "cam_$i"
done
for i in 1 2 3 4 5; do
    create_video train "vid$i"
done

# Process val
echo "=== Processing VAL ==="
for i in $(seq -w 1 25); do
    create_video val "cam_$i"
done
for i in 1 2 3 4 5; do
    create_video val "vid$i"
done

echo "=== All done ==="
echo "Videos saved to: $OUTPUT_DIR"
