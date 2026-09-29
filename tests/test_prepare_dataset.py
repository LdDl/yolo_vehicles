"""Regression checks for source annotation formats and dataset splits."""

import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_dataset as prep


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def jpeg(self, path, size=(800, 450), color=(20, 30, 40)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)

    def junction(self, rows):
        source = self.root / "junction-raw"
        source.mkdir()
        with (source / "sample_labels.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
            writer.writerows(rows)
        for index, name in enumerate(sorted({r[0] for r in rows})):
            self.jpeg(source / "sampled_images" / name, color=(index, 30, 40))
        return source

    def mio(self, rows):
        source = self.root / "mio-raw"
        source.mkdir()
        with (source / "gt_train.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        for index, name in enumerate(sorted({r[0] for r in rows})):
            self.jpeg(source / "train" / f"{name}.jpg", color=(index, 30, 40))
        return source

    def test_junction_uses_real_dimensions_and_our_mapping(self):
        source = self.junction([
            ["1112-5916_frame_1.jpg", 1280, 720, "taksi", 400, 225, 800, 450],
            ["1112-5916_frame_1.jpg", 1280, 720, "van", 0, 0, 80, 45],
            ["1112-5916_frame_1.jpg", 1280, 720, "person", 20, 20, 40, 40],
            ["1112-5917_frame_2.jpg", 1280, 720, "bicycle", 20, 20, 40, 40],
        ])
        output = self.root / "converted"
        stats = prep.convert_junction(source, output)
        lines = (output / "labels/1112-5916_frame_1.txt").read_text().splitlines()
        self.assertIn("0 0.7500000000 0.7500000000 0.5000000000 0.5000000000", lines)
        self.assertEqual(len(lines), 2)
        self.assertEqual((output / "labels/1112-5917_frame_2.txt").read_text(), "")
        self.assertEqual(stats["csv_size_mismatch_images"], 2)
        self.assertEqual(stats["empty_images"], 1)

    def test_mio_preserves_ids_and_excludes_ambiguous_invalid_and_test(self):
        source = self.mio([
            ["00000000", "car", 1, 2, 50, 80],
            ["00000001", "motorized_vehicle", 2, 3, 60, 90],
            ["00000001", "bus", 2, 3, 60, 90],
            ["00000002", "pickup_truck", -10, 0, 900, 450],
            ["00000003", "car", 1, 2, 1, 30],
        ])
        self.jpeg(source / "test/00110000.jpg")
        (source / "your_result_test.csv").write_text("00110000,car,1,2,3,4\n")
        output = self.root / "converted"
        stats = prep.convert_mio(source, output)
        self.assertEqual({p.name for p in (output / "images").iterdir()}, {"00000000.jpg", "00000002.jpg"})
        self.assertEqual((output / "labels/00000002.txt").read_text(), "3 0.5000000000 0.5000000000 1.0000000000 1.0000000000\n")
        self.assertEqual(stats["excluded_ambiguous_images"], 1)
        self.assertEqual(stats["excluded_invalid_box_images"], 1)
        self.assertEqual(stats["excluded_unlabelled_official_test_images"], 1)
        self.assertEqual(stats["class_objects"]["bus"], 0)

    def test_unknown_class_and_unannotated_images_fail(self):
        source = self.mio([["00000000", "hovercraft", 1, 2, 50, 80]])
        with self.assertRaisesRegex(ValueError, "unknown source classes"):
            prep.convert_mio(source, self.root / "output")
        self.assertFalse((self.root / "output").exists())
        self.jpeg(source / "train/00000001.jpg")
        with self.assertRaisesRegex(ValueError, "CSV/image mismatch"):
            prep.convert_mio(source, self.root / "output")

    def test_dry_run_and_existing_output(self):
        source = self.mio([["00000000", "car", 1, 2, 50, 80]])
        output = self.root / "output"
        prep.convert_mio(source, output, dry_run=True)
        self.assertFalse(output.exists())
        output.mkdir()
        (output / "keep").write_text("untouched")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prep.convert_mio(source, output)
        self.assertEqual((output / "keep").read_text(), "untouched")

    def test_source_groups_include_variants(self):
        self.assertEqual(prep.junction_group("aug_1112-5925_bilisimFatihframe_000100_182327.jpg"), "1112-5925")
        self.assertEqual(prep.junction_group("1112-5925_atakoy1_frame_1.jpg"), "1112-5925")
        self.assertEqual(prep.junction_group("aug_fuatOld-ch01_snapshot_123.jpg"), "fuatOld")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            prep.junction_group("unknown.jpg")

    def converted(self, name, specs):
        root = self.root / name
        records = []
        for image_name, group, color, augmented, label in specs:
            self.jpeg(root / "images" / image_name, color=color)
            (root / "labels").mkdir(exist_ok=True)
            label_name = Path(image_name).stem + ".txt"
            (root / "labels" / label_name).write_text(label)
            records.append({
                "image": f"images/{image_name}", "label": f"labels/{label_name}",
                "source_image": f"{name}/{image_name}", "source_dataset": name,
                "group": group, "augmented": augmented,
                "sha256": prep.file_hash(root / "images" / image_name),
            })
        prep.write_jsonl(root / "manifest.jsonl", records)
        prep.write_json(root / "metadata.json", {"format_version": 1, "names": prep.CLASS_NAMES})
        return root

    def test_merge_names_duplicates_groups_and_reproducibility(self):
        label = "0 0.5 0.5 0.2 0.2\n"
        specs = [(f"{i}.jpg", str(i // 2), (i * 11, 50, 70), False, label) for i in range(20)]
        first = self.converted("a", specs)
        second = self.converted("b", [
            ("0.jpg", "other", (250, 100, 100), False, label),
            ("duplicate.jpg", "linked", (0, 50, 70), False, label),
        ])
        summaries = []
        outputs = []
        for name in ("merged-1", "merged-2"):
            output = self.root / name
            summaries.append(prep.merge_datasets([first, second], output, seed=7))
            outputs.append([json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()])
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(summaries[0]["removed_exact_duplicates"], 1)
        self.assertEqual(sum(v["images"] for v in summaries[0]["splits"].values()), 21)
        self.assertEqual(len({r["image"] for r in outputs[0]}), 21)
        for group in {r["split_group"] for r in outputs[0]}:
            self.assertEqual(len({r["split"] for r in outputs[0] if r["split_group"] == group}), 1)
        for record in outputs[0]:
            self.assertTrue((self.root / "merged-1" / record["label"]).is_file())

    def test_held_out_augmentations_never_return_to_train(self):
        records = []
        for i in range(12):
            for augmented in (False, True):
                records.append((f"{'aug_' if augmented else ''}{i}.jpg", str(i), (i * 15, 80 + 50 * augmented, 20), augmented, "0 0.5 0.5 0.2 0.2\n"))
        source = self.converted("source", records)
        output = self.root / "merged"
        summary = prep.merge_datasets([source], output, ratios=(0.5, 0.25, 0.25))
        manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
        self.assertGreater(summary["excluded_reasons"]["augmented_image_in_held_out_group"], 0)
        self.assertTrue(all(not r["augmented"] for r in manifest if r["split"] != "train"))
        for group in {r["split_group"] for r in manifest}:
            self.assertEqual(len({r["split"] for r in manifest if r["split_group"] == group}), 1)

    def test_conflicting_labels_and_tampering(self):
        source = self.converted("source", [
            ("a.jpg", "a", (10, 20, 30), False, "0 0.5 0.5 0.2 0.2\n"),
            ("b.jpg", "b", (10, 20, 30), False, "3 0.5 0.5 0.2 0.2\n"),
            ("c.jpg", "c", (90, 60, 30), False, "0 0.5 0.5 0.2 0.2\n"),
        ])
        summary = prep.merge_datasets([source], self.root / "merged", ratios=(1, 0, 0))
        self.assertEqual(summary["excluded_reasons"]["identical_image_conflicting_labels"], 2)
        self.assertEqual(summary["splits"]["train"]["images"], 1)
        (source / "images/c.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image changed"):
            prep.merge_datasets([source], self.root / "another", ratios=(1, 0, 0))

    def test_download_skips_folder_or_archive(self):
        spec = prep.DATASETS["junction"]
        raw = self.root / "raw"
        folder = raw / spec["folder"]
        (folder / "sampled_images").mkdir(parents=True)
        (folder / "sample_labels.csv").write_text("sample")
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be called")):
            prep.download_dataset(raw, "junction")
            with zipfile.ZipFile(raw / spec["archive"], "w") as archive:
                archive.writestr(spec["folder"] + "/sample_labels.csv", "sample")
                archive.writestr(spec["folder"] + "/sampled_images/", "")
            shutil.rmtree(folder)
            prep.download_dataset(raw, "junction")
            self.assertEqual((folder / "sample_labels.csv").read_text(), "sample")

    def test_tar_extract_and_unsafe_zip(self):
        spec = prep.DATASETS["mio"]
        raw = self.root / "raw"
        raw.mkdir()
        archive_path = raw / spec["archive"]
        with tarfile.open(archive_path, "w") as archive:
            directory = tarfile.TarInfo(spec["folder"] + "/train")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            entry = tarfile.TarInfo(spec["folder"] + "/gt_train.csv")
            entry.size = 4
            archive.addfile(entry, io.BytesIO(b"test"))
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be called")):
            prep.download_dataset(raw, "mio")
        self.assertTrue((raw / spec["folder"] / "gt_train.csv").is_file())
        bad = raw / "bad.zip"
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr("../escape", "bad")
        with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
            prep.extract_archive(bad, raw, prep.DATASETS["junction"])
        self.assertFalse((self.root / "escape").exists())

    def test_download_missing_archive_and_publish_complete_file(self):
        spec = prep.DATASETS["junction"]
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(spec["folder"] + "/sample_labels.csv", "sample")
            archive.writestr(spec["folder"] + "/sampled_images/", "")
        response = io.BytesIO(payload.getvalue())
        response.headers = {"Content-Length": str(len(payload.getvalue()))}
        with patch("urllib.request.urlopen", return_value=response) as network:
            prep.download_dataset(self.root, "junction")
        network.assert_called_once()
        self.assertEqual((self.root / spec["archive"]).read_bytes(), payload.getvalue())
        self.assertFalse((self.root / (spec["archive"] + ".part")).exists())

    def test_incomplete_folder_does_not_trigger_download(self):
        spec = prep.DATASETS["mio"]
        (self.root / spec["folder"]).mkdir()
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be called")):
            with self.assertRaisesRegex(ValueError, "Incomplete dataset"):
                prep.download_dataset(self.root, "mio")

    def test_all_cli_runs_full_pipeline_with_existing_sources(self):
        junction = self.junction([
            [f"1112-{5900 + i}_frame_1.jpg", 1280, 720, "taksi", 40, 30, 100, 90]
            for i in range(6)
        ])
        mio = self.mio([[f"{i:08d}", "bus", 20, 30, 140, 160] for i in range(6)])
        for i in range(6):
            self.jpeg(mio / "train" / f"{i:08d}.jpg", color=(100 + i * 20, 60, 80))
        raw = self.root / "raw"
        raw.mkdir()
        junction.rename(raw / prep.DATASETS["junction"]["folder"])
        mio.rename(raw / prep.DATASETS["mio"]["folder"])
        output = self.root / "work"
        output.mkdir()
        (output / "keep.txt").write_text("existing working directory")
        training = self.root / "data/generated"
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be called")):
            prep.main([
                "all", "--directory", str(raw), "--output", str(output),
                "--ratios", "0.5", "0.25", "0.25", "--seed", "17", "--mode", "hardlink",
                "--training-dir", str(training), "--backup", str(self.root / "weights"),
            ])
        self.assertTrue((output / "prepared/junction/manifest.jsonl").is_file())
        self.assertTrue((output / "prepared/mio/manifest.jsonl").is_file())
        self.assertTrue((output / "merged/data.yaml").is_file())
        self.assertTrue((training / "vehicles.yaml").is_file())
        self.assertIn(f"names = {training}/vehicles.names\n", (training / "vehicles.data").read_text())
        self.assertFalse((training.parent / "vehicles.yaml").exists())
        summary = json.loads((output / "merged/summary.json").read_text())
        self.assertEqual(summary["seed"], 17)
        self.assertEqual(summary["requested_ratios"], {"train": 0.5, "val": 0.25, "test": 0.25})
        self.assertEqual(sum(v["images"] for v in summary["splits"].values()), 12)
        for split in prep.SPLITS:
            self.assertGreater(summary["splits"][split]["images"], 0)
        record = json.loads((output / "merged/manifest.jsonl").read_text().splitlines()[0])
        self.assertEqual((output / "merged" / record["image"]).stat().st_ino, Path(record["source_image"]).stat().st_ino)
        self.assertEqual((output / "keep.txt").read_text(), "existing working directory")

    def test_all_rejects_existing_output_before_downloading(self):
        for relative in ("prepared/junction", "prepared/mio", "merged"):
            with self.subTest(relative=relative):
                output = self.root / relative.replace("/", "-")
                (output / relative).mkdir(parents=True)
                with patch.object(prep, "download_dataset") as download:
                    with self.assertRaisesRegex(ValueError, "Output already exists"):
                        prep.prepare_all(self.root / "raw", output)
                    download.assert_not_called()

    def test_all_rejects_invalid_ratios_before_downloading(self):
        with patch.object(prep, "download_dataset") as download:
            with self.assertRaisesRegex(ValueError, "sum to 1"):
                prep.prepare_all(self.root / "raw", self.root / "output", ratios=(0.8, 0.2, 0.2))
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
