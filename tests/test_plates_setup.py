"""Plate preparation and training integration without downloading or training models."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
import yaml

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("plates_prepare", PROJECT / "plates/prepare_dataset.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
sys.path.insert(0, str(PROJECT / "scripts"))
import train_ultralytics as ultra


class PlateSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        for index, split in enumerate(prep.SPLITS):
            images = self.dataset / "images" / split
            labels = self.dataset / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (8, 8), (index * 90, 0, 0)).save(images / "0.png")
            (labels / "0.txt").write_text("" if split == "test" else "4 0.5 0.5 0.25 0.25\n")
        (self.dataset / "data_multiclass.yaml").write_text(yaml.safe_dump({"names": prep.NAMES}))

    def test_configure_preserves_fifth_class_background_and_split(self):
        output = self.root / "generated"
        prep.configure(self.dataset, output, self.root / "weights")
        config = yaml.safe_load((output / "plates.yaml").read_text())
        self.assertEqual(prep.NAMES, ["civilian", "taxi", "military", "police", "diplomatic"])
        self.assertEqual((output / "plates.names").read_bytes(), (PROJECT / "plates/classes.names").read_bytes())
        self.assertEqual(config["names"][4], "diplomatic")
        self.assertEqual((self.dataset / "images/test/0.txt").read_text(), "")
        self.assertIn("plates-test.txt", (output / "plates-test.data").read_text())
        self.assertEqual(len((output / "plates-train.txt").read_text().splitlines()), 1)
        self.assertFalse((output / "vehicles.yaml").exists())
        prep.configure(self.dataset, output, self.root / "weights")

    def test_invalid_labels_and_cross_split_duplicates_fail(self):
        path = self.dataset / "labels/train/0.txt"
        original = path.read_text()
        for bad in ["5 0.5 0.5 0.2 0.2\n", "0 nan 0.5 0.2 0.2\n", "0 0.99 0.5 0.2 0.2\n"]:
            path.write_text(bad)
            with self.assertRaises(ValueError):
                prep.validate(self.dataset)
        path.write_text(original)
        shutil.copy2(self.dataset / "images/train/0.png", self.dataset / "images/val/0.png")
        with self.assertRaisesRegex(ValueError, "across splits"):
            prep.validate(self.dataset)

    def test_archive_extract_and_path_traversal(self):
        archive = self.root / "dataset.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for path in self.dataset.rglob("*"):
                if path.is_file():
                    z.write(path, "wrapper/" + path.relative_to(self.dataset).as_posix())
        target = self.root / "downloaded"
        prep.extract(archive, target)
        self.assertTrue((target / "images/train/0.png").is_file())
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("../escape.txt", "escape")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            prep.extract(archive, self.root / "unsafe")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_plate_darknet_templates_and_launcher(self):
        import re
        project = self.root / "project"
        for folder in ["scripts", "plates/configs", "data/generated", "bin"]:
            (project / folder).mkdir(parents=True)
        for v in [3, 4]:
            train = PROJECT / f"plates/configs/yolov{v}-tiny-plates.cfg"
            infer = train.with_name(f"yolov{v}-tiny-plates-infer.cfg")
            strip = lambda s: re.sub(r"(?m)^(batch|subdivisions|random)=.*$", "", s)
            self.assertEqual(strip(train.read_text()), strip(infer.read_text()))
            self.assertIn("width=320\nheight=192", train.read_text())
            self.assertEqual(train.read_text().count("\nclasses=5\n"), 2)
            self.assertEqual(train.read_text().count("\nfilters=30\n"), 2)
            shutil.copy2(train, project / "plates/configs" / train.name)
            shutil.copy2(infer, project / "plates/configs" / infer.name)
        script = project / "scripts/train_darknet.sh"
        shutil.copy2(PROJECT / "scripts/train_darknet.sh", script)
        (project / "data/generated/plates.data").write_text("classes = 5\nbackup = old\n")
        binary = project / "bin/darknet"
        binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
        binary.chmod(0o755)
        env = {**os.environ, "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"]}
        result = subprocess.run(["bash", str(script), "v4-tiny", "--task", "plates", "--scratch"], env=env, capture_output=True, text=True, check=True)
        self.assertIn("plates/configs/yolov4-tiny-plates.cfg", result.stdout)
        self.assertIn("-clear", result.stdout)
        self.assertTrue((project / "weights/yolov4-tiny-plates/yolov4-tiny-plates-infer.cfg").is_file())

    def test_ultralytics_plates_selects_data_augmentations_and_export(self):
        calls = []
        best = self.root / "best.pt"
        best.write_bytes(b"fixture")

        class FakeYOLO:
            def __init__(self, path):
                self.trainer = types.SimpleNamespace(best=best)

            def train(self, **kw):
                calls.append(kw)

            def export(self, **kw):
                calls.append(kw)

        generated = self.root / "data/generated"
        generated.mkdir(parents=True)
        (generated / "plates.yaml").write_text("fixture")
        with patch.object(ultra, "PROJECT", self.root), patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=FakeYOLO)}):
            ultra.main(["--task", "plates", "--model", "v8n", "--scratch"])
        self.assertEqual(calls[0]["name"], "yolov8n-plates")
        self.assertEqual(calls[0]["data"], str(generated / "plates.yaml"))
        self.assertEqual(calls[0]["imgsz"], 320)
        self.assertEqual(calls[0]["fliplr"], 0)
        self.assertEqual(calls[0]["hsv_h"], 0)
        self.assertEqual(calls[1]["imgsz"], [192, 320])


if __name__ == "__main__":
    unittest.main()
