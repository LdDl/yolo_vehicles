"""Exercise OCR dataset preparation without network access or model execution."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
import yaml


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ocr_dataset_prepare", PROJECT / "ocr/prepare_dataset.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class OCRDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        for index, split in enumerate(prep.SPLITS):
            images = self.dataset / "images" / split
            labels = self.dataset / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (20, 8), (index * 70, 40, 20)).save(images / "crop.png")
            (labels / "crop.txt").write_text("" if split == "test" else "21 0.5 0.5 0.2 0.5\n")
        config = {split: f"images/{split}" for split in prep.SPLITS}
        config.update(nc=23, names=dict(enumerate(prep.NAMES)))
        (self.dataset / "data.yaml").write_text(yaml.safe_dump(config))
        (self.dataset / "classes.names").write_text("\n".join(prep.NAMES) + "\n")

    def archive(self, destination, prefix=""):
        with zipfile.ZipFile(destination, "w") as stream:
            for path in self.dataset.rglob("*"):
                if path.is_file():
                    stream.write(path, prefix + path.relative_to(self.dataset).as_posix())

    def manifest(self, shared_group=False):
        with (self.dataset / "manifest.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "label", "group", "sha256"])
            for index, split in enumerate(prep.SPLITS):
                image = self.dataset / f"images/{split}/crop.png"
                writer.writerow([f"images/{split}/crop.png", f"labels/{split}/crop.txt", 0 if shared_group else index,
                                 hashlib.sha256(image.read_bytes()).hexdigest()])

    def test_generated_files_preserve_backgrounds_absent_d_and_other_tasks(self):
        self.manifest()
        output = self.root / "generated"
        output.mkdir()
        for name in ("plates.yaml", "vehicles.data"):
            (output / name).write_text("existing task")
        summary = prep.configure(self.dataset, output, self.root / "weights")
        self.assertEqual(prep.NAMES, (PROJECT / "ocr/classes.names").read_text().split())
        config = yaml.safe_load((output / "ocr.yaml").read_text())
        self.assertEqual(config["names"][22], "D")
        self.assertEqual(config["names"][0], "0")
        self.assertEqual(config["nc"], 23)
        self.assertEqual(config["path"], str(self.dataset))
        self.assertIn("classes = 23\n", (output / "ocr.data").read_text())
        self.assertIn(f"valid = {output}/ocr-test.txt", (output / "ocr-test.data").read_text())
        self.assertEqual((self.dataset / "images/test/crop.txt").read_bytes(), b"")
        for split in prep.SPLITS:
            self.assertEqual(summary[split]["class_objects"]["D"], 0)
            self.assertEqual((output / f"ocr-{split}.txt").read_text(), str(self.dataset / f"images/{split}/crop.png") + "\n")
        self.assertEqual(summary["test"]["empty_images"], 1)
        self.assertEqual(prep.configure(self.dataset, output, self.root / "weights"), summary)
        for name in ("plates.yaml", "vehicles.data"):
            self.assertEqual((output / name).read_text(), "existing task")

    def test_d_annotations_are_supported_when_examples_are_added(self):
        (self.dataset / "labels/train/crop.txt").write_text("22 0.5 0.5 0.2 0.5\n")
        _, summary = prep.validate(self.dataset)
        self.assertEqual(summary["train"]["class_objects"]["D"], 1)

    def test_missing_background_label_is_not_silently_created(self):
        (self.dataset / "labels/test/crop.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing pairs"):
            prep.configure(self.dataset, self.root / "generated", self.root / "weights")
        self.assertFalse((self.root / "generated").exists())

    def test_invalid_boxes_ids_and_taxonomy_are_rejected(self):
        label = self.dataset / "labels/train/crop.txt"
        original = label.read_text()
        for bad in ("23 0.5 0.5 0.2 0.5", "-1 0.5 0.5 0.2 0.5", "0 nan 0.5 0.2 0.5", "0 0.99 0.5 0.2 0.5", "0 0.5 0.5 0 0.5"):
            with self.subTest(label=bad):
                label.write_text(bad + "\n")
                with self.assertRaises(ValueError):
                    prep.validate(self.dataset)
        label.write_text(original)
        config = yaml.safe_load((self.dataset / "data.yaml").read_text())
        config["names"][21], config["names"][22] = "D", "Y"
        (self.dataset / "data.yaml").write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "published order"):
            prep.validate(self.dataset)

    def test_adjacent_annotation_conflicts_fail_before_writing(self):
        adjacent = self.dataset / "images/train/crop.txt"
        adjacent.write_text("0 0.5 0.5 0.2 0.5\n")
        with self.assertRaisesRegex(ValueError, "Conflicting adjacent"):
            prep.configure(self.dataset, self.root / "generated", self.root / "weights")
        self.assertFalse((self.root / "generated").exists())
        self.assertTrue(adjacent.read_text().startswith("0 "))

    def test_exact_duplicates_across_splits_are_rejected(self):
        shutil.copy2(self.dataset / "images/train/crop.png", self.dataset / "images/val/crop.png")
        with self.assertRaisesRegex(ValueError, "duplicate across splits"):
            prep.validate(self.dataset)

    def test_manifest_groups_and_hashes_are_checked(self):
        self.manifest(shared_group=True)
        with self.assertRaisesRegex(ValueError, "group crosses splits"):
            prep.validate(self.dataset)
        self.manifest()
        path = self.dataset / "manifest.csv"
        text = path.read_text()
        digest = hashlib.sha256((self.dataset / "images/train/crop.png").read_bytes()).hexdigest()
        path.write_text(text.replace(digest, "0" * 64))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            prep.validate(self.dataset)

    def test_flat_wrapped_and_nested_archives(self):
        for mode in ("flat", "wrapped", "nested"):
            with self.subTest(mode=mode):
                archive = self.root / f"{mode}.zip"
                self.archive(archive, "wrapper/" if mode == "wrapped" else "")
                if mode == "nested":
                    inner = archive.read_bytes()
                    with zipfile.ZipFile(archive, "w") as stream:
                        stream.writestr("russian-license-plate-characters-23-classes_v1.zip", inner)
                destination = self.root / mode
                prep.extract(archive, destination)
                self.assertEqual((destination / "labels/test/crop.txt").read_bytes(), b"")
                self.assertTrue((destination / "data.yaml").is_file())

    def test_unsafe_archive_and_invalid_data_are_not_published(self):
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../escape.txt", "escape")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            prep.extract(archive, self.root / "downloaded")
        self.assertFalse((self.root / "escape.txt").exists())
        self.assertFalse((self.root / "downloaded").exists())
        (self.dataset / "labels/val/crop.txt").unlink()
        self.archive(archive)
        with self.assertRaisesRegex(ValueError, "Missing pairs"):
            prep.extract(archive, self.root / "downloaded")
        self.assertFalse((self.root / "downloaded").exists())

    def test_existing_archive_and_dataset_are_reused_without_network(self):
        archive = self.root / "existing.zip"
        self.archive(archive)
        with patch.object(prep.subprocess, "run") as command:
            prep.download(archive)
            command.assert_not_called()
        with patch.object(prep, "PROJECT", self.root), patch.object(prep, "download") as download:
            prep.main(["--dataset", str(self.dataset), "--output", str(self.root / "generated")])
            download.assert_not_called()

    def test_download_uses_published_kaggle_slug(self):
        fixture = self.root / "fixture.zip"
        self.archive(fixture)

        def fake_download(command, check):
            self.assertEqual(command[:5], ["kaggle", "datasets", "download", "-d", prep.SLUG])
            self.assertTrue(check)
            shutil.copy2(fixture, Path(command[-1]) / "download.zip")

        target = self.root / "raw/dataset.zip"
        with patch.object(prep.shutil, "which", return_value="/venv/bin/kaggle"), patch.object(prep.subprocess, "run", side_effect=fake_download):
            prep.download(target)
        self.assertEqual(target.read_bytes(), fixture.read_bytes())


if __name__ == "__main__":
    unittest.main()
