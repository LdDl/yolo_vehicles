#!/usr/bin/env python3
"""Download official initial weights for the public-dataset training runs."""

import argparse
from pathlib import Path
import shutil
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
DARKNET_REVISION = "master"
ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
MODELS = {
    "v3-tiny": {
        "yolov3-tiny.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
        "yolov3-tiny-coco.cfg": f"https://raw.githubusercontent.com/AlexeyAB/darknet/{DARKNET_REVISION}/cfg/yolov3-tiny.cfg",
    },
    "v4-tiny": {
        "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
        "yolov4-tiny-coco.cfg": f"https://raw.githubusercontent.com/AlexeyAB/darknet/{DARKNET_REVISION}/cfg/yolov4-tiny.cfg",
    },
    "v5nu": {"yolov5nu.pt": f"{ASSETS}/yolov5nu.pt"},
    "v5su": {"yolov5su.pt": f"{ASSETS}/yolov5su.pt"},
    "v5mu": {"yolov5mu.pt": f"{ASSETS}/yolov5mu.pt"},
    "v8n": {"yolov8n.pt": f"{ASSETS}/yolov8n.pt"},
    "v9t": {"yolov9t.pt": f"{ASSETS}/yolov9t.pt"},
    "v11n": {"yolo11n.pt": f"{ASSETS}/yolo11n.pt"},
}
DEFAULT_MODELS = ("v3-tiny", "v4-tiny", "v5nu", "v8n", "v9t", "v11n")


def download(url, destination):
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Exists: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vehicles-yolo/1.0"})
    with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as target:
        length = source.headers.get("Content-Length")
        shutil.copyfileobj(source, target)
    size = partial.stat().st_size
    if not size or (length is not None and size != int(length)):
        raise ValueError(f"Incomplete download: {partial}")
    with partial.open("rb") as stream:
        header = stream.read(256).lstrip().lower()
    if header.startswith((b"<!doctype", b"<html")):
        raise ValueError(f"Server returned HTML: {url}")
    partial.replace(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--output", type=Path, default=PROJECT / "weights/pretrained")
    args = parser.parse_args(argv)
    for name in DEFAULT_MODELS if args.model == "all" else [args.model]:
        for filename, url in MODELS[name].items():
            download(url, args.output.expanduser().resolve() / filename)


if __name__ == "__main__":
    main()
