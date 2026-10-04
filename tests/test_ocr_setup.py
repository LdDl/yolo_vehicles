"""Check OCR integration without training or loading model weights."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import train_ultralytics as ultra


class OCRSetupTests(unittest.TestCase):
    def test_darknet_heads_and_launchers_use_ocr_configs_and_outputs(self):
        names = (PROJECT / "ocr/classes.names").read_text().splitlines()
        self.assertEqual(names, list("0123456789ABCEHKMOPTXYD"))
        for version in (3, 4):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp)
                for directory in ("scripts", "ocr/configs", "data/generated", "bin"):
                    (project / directory).mkdir(parents=True)
                base = f"yolov{version}-tiny-ocr"
                train = PROJECT / f"ocr/configs/{base}.cfg"
                infer = PROJECT / f"ocr/configs/{base}-infer.cfg"
                strip = lambda text: re.sub(r"(?m)^(batch|subdivisions)=.*$", "", text)
                self.assertEqual(strip(train.read_text()), strip(infer.read_text()))
                self.assertIn("width=224\nheight=64", train.read_text())
                sections = []
                for line in train.read_text().splitlines():
                    if line.startswith("["):
                        sections.append({"type": line})
                    elif "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        sections[-1][key] = value
                heads = 0
                for index, section in enumerate(sections):
                    if section["type"] == "[yolo]":
                        heads += 1
                        self.assertEqual(int(section["classes"]), len(names))
                        self.assertEqual(int(sections[index - 1]["filters"]), (len(names) + 5) * 3)
                        self.assertEqual(section["random"], "0")
                self.assertEqual(heads, 2)
                shutil.copy2(train, project / "ocr/configs" / train.name)
                shutil.copy2(infer, project / "ocr/configs" / infer.name)
                script = project / "scripts/train_darknet.sh"
                shutil.copy2(PROJECT / "scripts/train_darknet.sh", script)
                (project / "data/generated/ocr.data").write_text("classes = 23\nbackup = old\n")
                binary = project / "bin/darknet"
                binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
                binary.chmod(0o755)
                env = {**os.environ, "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"]}
                command = ["bash", str(script), f"v{version}-tiny", "--task", "ocr", "--scratch"]
                result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                self.assertIn(str(project / f"ocr/configs/{base}.cfg"), result.stdout)
                run = project / "weights" / base
                self.assertTrue((run / infer.name).is_file())
                data = (project / f"data/generated/{base}.data").read_text()
                self.assertIn(f"backup = {run}", data)
                self.assertFalse((project / f"weights/yolov{version}-tiny-plates").exists())
                repeat = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertNotEqual(repeat.returncode, 0)
                self.assertIn("Run already exists", repeat.stderr)

    def test_ultralytics_ocr_defaults_for_all_four_models(self):
        for model in ("v5nu", "v8n", "v9t", "v11n"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generated = root / "data/generated"
                generated.mkdir(parents=True)
                (generated / "ocr.yaml").write_text("fixture")
                best = root / "best.pt"
                best.write_bytes(b"fixture")
                calls = []

                class FakeYOLO:
                    def __init__(self, path):
                        self.trainer = types.SimpleNamespace(best=best)

                    def train(self, **kwargs):
                        calls.append(kwargs)

                    def export(self, **kwargs):
                        calls.append(kwargs)

                with patch.object(ultra, "PROJECT", root), patch.dict(sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=FakeYOLO)}):
                    ultra.main(["--task", "ocr", "--model", model, "--scratch"])
                self.assertEqual(calls[0]["name"], f"{ultra.MODELS[model]}-ocr")
                self.assertEqual(calls[0]["data"], str(generated / "ocr.yaml"))
                self.assertEqual(calls[0]["imgsz"], 224)
                self.assertTrue(calls[0]["rect"])
                for key in ("fliplr", "flipud", "mosaic", "mixup", "degrees", "shear", "perspective"):
                    self.assertEqual(calls[0][key], 0)
                self.assertEqual(calls[1]["imgsz"], [64, 224])
                self.assertFalse(calls[1]["dynamic"])
                self.assertFalse(calls[1]["half"])


if __name__ == "__main__":
    unittest.main()
