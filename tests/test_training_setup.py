"""Check public-dataset integration using fixtures and mocked training processes."""

import contextlib
import io
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import prepare_dataset as prep
import train_ultralytics as ultra
import download_pretrained as download


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = io.StringIO()
        manager = contextlib.redirect_stdout(self.output)
        manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)

    def dataset(self):
        dataset = self.root / 'public merged'
        dataset.mkdir()
        (dataset / 'metadata.json').write_text(json.dumps({'names': prep.CLASS_NAMES}))
        for index, split in enumerate(prep.SPLITS):
            images, labels = dataset / 'images' / split, dataset / 'labels' / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / 'frame.jpg').write_bytes(b'image header is not decoded by configure')
            (labels / 'frame.txt').write_text('' if split == 'test' else f'{index} 0.5 0.5 0.2 0.2\n')
        return dataset

    def test_configure_pairs_background_and_relocation(self):
        dataset = self.dataset()
        original = {p: p.read_bytes() for p in dataset.rglob('*') if p.is_file()}
        output, backup = self.root / 'data', self.root / 'weights'
        prep.configure_training(dataset, output, backup, True)
        self.assertEqual((dataset / 'images/test/frame.txt').read_text(), '')
        self.assertIn(str(dataset / 'images/val/frame.jpg'), (output / 'vehicles-val.txt').read_text())
        self.assertIn('vehicles-test.txt', (output / 'vehicles-test.data').read_text())
        self.assertIn('vehicles-val.txt', (output / 'vehicles.data').read_text())
        self.assertTrue(all(p.read_bytes() == contents for p, contents in original.items()))
        moved = self.root / 'relocated'
        shutil.move(dataset, moved)
        prep.configure_training(moved, output, backup, True)
        import yaml
        self.assertEqual(yaml.safe_load((output / 'vehicles.yaml').read_text())['path'], str(moved))

    def test_configure_missing_and_conflicting_labels_fail_before_output(self):
        dataset = self.dataset()
        output = self.root / 'data'
        label = dataset / 'labels/train/frame.txt'
        contents = label.read_text()
        label.unlink()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            prep.configure_training(dataset, output, self.root / 'weights', True)
        self.assertFalse(output.exists())
        label.write_text(contents)
        (dataset / 'images/train/frame.txt').write_text('different annotation')
        with self.assertRaisesRegex(ValueError, 'Conflicting adjacent'):
            prep.configure_training(dataset, output, self.root / 'weights', True)
        self.assertFalse(output.exists())

    def test_configure_cli_defaults_to_generated_without_touching_legacy_files(self):
        dataset = self.dataset()
        project = self.root / 'project'
        legacy = project / 'data'
        legacy.mkdir(parents=True)
        originals = {}
        for name in ('vehicles.yaml', 'vehicles.data', 'vehicles.names'):
            path = legacy / name
            path.write_text(f'legacy file: {name}\n')
            originals[path] = path.read_bytes()
        subprocess.run([
            sys.executable, str(PROJECT / 'scripts/prepare_dataset.py'),
            'configure', '--dataset', str(dataset), '--darknet-labels',
        ], cwd=project, check=True, capture_output=True)
        generated = project / 'data/generated'
        self.assertTrue(all(path.read_bytes() == content for path, content in originals.items()))
        self.assertEqual((generated / 'vehicles.names').read_text().splitlines(), prep.CLASS_NAMES)
        for split in prep.SPLITS:
            self.assertEqual((generated / f'vehicles-{split}.txt').read_text(), f'{dataset}/images/{split}/frame.jpg\n')
        for suffix, split in (('', 'val'), ('-test', 'test')):
            config = (generated / f'vehicles{suffix}.data').read_text()
            self.assertIn(f'train = {generated}/vehicles-train.txt\n', config)
            self.assertIn(f'valid = {generated}/vehicles-{split}.txt\n', config)
            self.assertIn(f'names = {generated}/vehicles.names\n', config)
        import yaml
        self.assertEqual(yaml.safe_load((generated / 'vehicles.yaml').read_text())['path'], str(dataset))

    def test_ultralytics_download_train_export_resume_and_scratch(self):
        data = self.root / 'data/generated/vehicles.yaml'
        prep.configure_training(self.dataset(), data.parent, self.root / 'weights')
        legacy = self.root / 'data/vehicles.yaml'
        legacy.write_text('legacy config must not be used')
        calls = []

        class FakeYOLO:
            def __init__(self, path):
                self.path = path
                calls.append(('load', path))

            def train(self, **kwargs):
                calls.append(('train', kwargs))
                run = Path(self.path).parent.parent if kwargs.get('resume') else Path(kwargs['project']) / kwargs['name']
                best = run / 'weights/best.pt'
                best.parent.mkdir(parents=True, exist_ok=True)
                best.write_bytes(b'fake checkpoint, no training')
                self.trainer = types.SimpleNamespace(best=best)

            def export(self, **kwargs):
                calls.append(('export', kwargs))

        def fake_download(url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b'fake initial weights, no download')

        cases = [
            ('v5nu', 'yolov5nu', 'yolov5n.yaml'),
            ('v5su', 'yolov5su', 'yolov5s.yaml'),
            ('v5mu', 'yolov5mu', 'yolov5m.yaml'),
            ('v8n', 'yolov8n', 'yolov8n.yaml'),
            ('v9t', 'yolov9t', 'yolov9t.yaml'),
            ('v11n', 'yolo11n', 'yolo11n.yaml'),
        ]
        fake = types.SimpleNamespace(YOLO=FakeYOLO)
        pretrained = self.root / 'weights/pretrained'
        with patch.dict(sys.modules, {'ultralytics': fake}), patch.object(ultra, 'PROJECT', self.root), patch.object(download, 'download', side_effect=fake_download) as fetch:
            download.main(['--model', 'all', '--output', str(pretrained)])
            self.assertTrue((pretrained / 'yolov5nu.pt').is_file())
            self.assertFalse((pretrained / 'yolov5n.pt').exists())
            for key, name, architecture in cases:
                with self.subTest(model=key):
                    calls.clear()
                    download.main(['--model', key, '--output', str(pretrained)])
                    weights = pretrained / f'{name}.pt'
                    fetch.assert_called_with(f'{download.ASSETS}/{name}.pt', weights)
                    ultra.main(['--model', key, '--output', str(self.root / 'runs')])
                    self.assertEqual(calls[0], ('load', str(weights)))
                    train_call = [kw for action, kw in calls if action == 'train'][-1]
                    self.assertFalse(train_call['exist_ok'])
                    self.assertTrue(train_call['pretrained'])
                    self.assertEqual(train_call['data'], str(data))
                    export = [kw for action, kw in calls if action == 'export'][-1]
                    self.assertEqual(export['format'], 'onnx')
                    self.assertEqual(export['imgsz'], [256, 416])
                    self.assertEqual(export['batch'], 1)
                    self.assertIsNone(export['nms'])
                    self.assertFalse(export['dynamic'])
                    self.assertFalse(export['half'])
                    best = self.root / f'runs/{name}-vehicles/weights/best.pt'
                    self.assertEqual(calls[-2], ('load', str(best)))
                    ultra.main(['--resume', str(best)])
                    self.assertEqual([kw for action, kw in calls if action == 'train'][-1], {'resume': True})
                    calls.clear()
                    weights.unlink()
                    ultra.main(['--model', key, '--scratch', '--output', str(self.root / 'scratch')])
                    self.assertEqual(calls[0], ('load', architecture))
                    train_call = [kw for action, kw in calls if action == 'train'][-1]
                    self.assertFalse(train_call['pretrained'])

    def test_downloader_skips_completed_files_and_rejects_html(self):
        target = self.root / 'initial.pt'
        target.write_bytes(b'kept')
        with patch.object(download.urllib.request, 'urlopen', side_effect=AssertionError('Network forbidden')):
            download.download('https://example.invalid/weights', target)
        self.assertEqual(target.read_bytes(), b'kept')
        target.unlink()
        response = io.BytesIO(b'<html>not weights</html>')
        response.headers = {}
        with patch.object(download.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'HTML'):
                download.download('https://example.invalid/weights', target)
        self.assertFalse(target.exists())

    def test_darknet_launcher_initialization_and_resume_without_training(self):
        project = self.root / 'project'
        for folder in ['scripts', 'configs', 'data/generated', 'weights/pretrained', 'bin']:
            (project / folder).mkdir(parents=True)
        script = project / 'scripts/train_darknet.sh'
        shutil.copy2(PROJECT / 'scripts/train_darknet.sh', script)
        for suffix in ['', '-infer']:
            (project / f'configs/yolov3-tiny-vehicles{suffix}.cfg').write_text('[net]\nwidth=416\nheight=256\n')
        (project / 'data/generated/vehicles.data').write_text('train = public-train\nvalid = public-val\nbackup = old\n')
        legacy = project / 'data/vehicles.data'
        legacy.write_text('legacy config must not be used')
        (project / 'weights/pretrained/yolov3-tiny.weights').write_bytes(b'fixture')
        (project / 'weights/pretrained/yolov3-tiny-coco.cfg').write_text('fixture')
        binary = project / 'bin/darknet'
        binary.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ['TEST_DARKNET_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'partial':
    Path(sys.argv[4]).write_bytes(b'partial fixture')
''')
        binary.chmod(0o755)
        env = os.environ.copy()
        env['PATH'] = str(binary.parent) + os.pathsep + env['PATH']
        log = project / 'commands.jsonl'
        env['TEST_DARKNET_LOG'] = str(log)
        subprocess.run(['bash', str(script), 'v3-tiny'], env=env, check=True, capture_output=True)
        commands = [json.loads(row) for row in log.read_text().splitlines()]
        self.assertEqual(commands[0][-1], '11')
        self.assertIn('-clear', commands[1])
        self.assertEqual(commands[1][2], str(project / 'data/generated/yolov3-tiny-vehicles.data'))
        self.assertIn('train = public-train\n', Path(commands[1][2]).read_text())
        self.assertEqual(legacy.read_text(), 'legacy config must not be used')
        run = project / 'weights/yolov3-tiny-vehicles'
        best = run / 'yolov3-tiny-vehicles_best.weights'
        best.write_bytes(b'previous best')
        last = run / 'yolov3-tiny-vehicles_last.weights'
        last.write_bytes(b'last')
        subprocess.run(['bash', str(script), 'v3-tiny', '--resume', str(last)], env=env, check=True, capture_output=True)
        commands = [json.loads(row) for row in log.read_text().splitlines()]
        self.assertNotIn('-clear', commands[-1])
        self.assertEqual(next(run.glob('*before-resume*.weights')).read_bytes(), b'previous best')
        self.assertIn(str(run), (project / 'data/generated/yolov3-tiny-vehicles.data').read_text())

    def test_darknet_train_and_infer_templates_match_architecture(self):
        import re
        for version in (3, 4):
            train = (PROJECT / f'configs/yolov{version}-tiny-vehicles.cfg').read_text()
            infer = (PROJECT / f'configs/yolov{version}-tiny-vehicles-infer.cfg').read_text()
            strip = lambda s: re.sub(r'(?m)^(batch|subdivisions|random)=.*$', '', s)
            self.assertEqual(strip(train), strip(infer))
            self.assertIn('width=416\nheight=256', train)
            self.assertIn('letter_box=1', train)
            self.assertIn('mosaic=0', train)
            self.assertEqual(len(re.findall(r'(?m)^classes=4$', train)), 2)
            self.assertEqual(len(re.findall(r'(?m)^filters=27$', train)), 2)


if __name__ == '__main__':
    unittest.main()
