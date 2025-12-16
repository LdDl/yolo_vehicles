#!/bin/bash
# Split existing distilled_data into train/val sets

DISTILLED_DIR="${1:-distilled_data}"
VAL_FRACTION="${2:-0.1}"

if [ ! -d "$DISTILLED_DIR/labels" ]; then
    echo "Error: $DISTILLED_DIR/labels not found"
    exit 1
fi

cd "$DISTILLED_DIR"

# Count total files
total=$(ls labels/*.txt 2>/dev/null | wc -l)
if [ "$total" -eq 0 ]; then
    echo "No label files found"
    exit 1
fi

# Calculate val count
val_count=$(echo "$total * $VAL_FRACTION" | bc | cut -d. -f1)
train_count=$((total - val_count))

echo "Total: $total files"
echo "Val: $val_count files (${VAL_FRACTION})"
echo "Train: $train_count files"
echo

mkdir -p train/images train/labels val/images val/labels

# Get shuffled list, take val_count for validation
ls labels/*.txt | shuf | head -n "$val_count" > val_list.txt

# Move val files
while read f; do
    name=$(basename "$f" .txt)
    mv "labels/$name.txt" val/labels/
    mv "images/$name.jpg" val/images/ 2>/dev/null || true
    mv "images/$name.png" val/images/ 2>/dev/null || true
done < val_list.txt

# Move rest to train
mv labels/*.txt train/labels/ 2>/dev/null || true
mv images/*.jpg train/images/ 2>/dev/null || true
mv images/*.png train/images/ 2>/dev/null || true

rm val_list.txt
rmdir labels images 2>/dev/null || true

echo "Done!"
echo "  train/images: $(ls train/images | wc -l) files"
echo "  train/labels: $(ls train/labels | wc -l) files"
echo "  val/images: $(ls val/images | wc -l) files"
echo "  val/labels: $(ls val/labels | wc -l) files"
