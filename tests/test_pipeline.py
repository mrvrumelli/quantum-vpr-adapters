import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from vpr_baseline.cache import load_cache
from vpr_baseline.cli import main
from vpr_baseline.data import SPLITS, load_manifests
from vpr_baseline.extract import extract_descriptors
from vpr_baseline.model import DESCRIPTOR_DIM, load_cosplace


class ToyModel(nn.Module):
    def __init__(self, dim=DESCRIPTOR_DIM):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(()))
        self.inference_modes = []

    def forward(self, images):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        colors = images.mean(dim=(2, 3))
        return nn.functional.pad(colors, (0, self.dim - colors.shape[1])) * self.scale


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        for split in SPLITS:
            folder = self.root / split
            folder.mkdir()
            for name, color in (("red", "red"), ("blue", "blue"), ("query", "red")):
                Image.new("RGB", (59, 42), color).save(folder / f"{name}.png")
            (folder / "gallery.csv").write_text(
                f"image_id,path\n{split}-blue,{split}/blue.png\n{split}-red,{split}/red.png\n", encoding="utf-8"
            )
            (folder / "query.csv").write_text(
                f"image_id,path\n{split}-query,{split}/query.png\n", encoding="utf-8"
            )
        (self.root / "val/positives.csv").write_text("query_id,gallery_id\nval-query,val-red\n", encoding="utf-8")
        (self.root / "test/positives.csv").write_text("INVALID TEST LABEL FILE\n", encoding="utf-8")

    def run_cli(self, command, split, *options):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main([
                command, "--dataset-root", str(self.root), "--cache-dir", str(self.cache),
                "--split", split, *options,
            ])
        self.assertEqual(result, 0)
        return json.loads(output.getvalue())

    def test_extraction_shape_order_and_frozen_inference(self):
        model = ToyModel()
        images = load_manifests(self.root)["train"]["gallery"]
        vectors = extract_descriptors(model, self.root, images, DESCRIPTOR_DIM, image_size=(40, 64), batch_size=1)
        self.assertEqual(vectors.shape, (2, DESCRIPTOR_DIM))
        torch.testing.assert_close(vectors.norm(dim=1), torch.ones(2))
        self.assertEqual(vectors[:, :3].argmax(dim=1).tolist(), [2, 0])
        self.assertEqual(model.inference_modes, [True, True])
        self.assertFalse(model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_extraction_rejects_wrong_model_dimension(self):
        images = load_manifests(self.root)["train"]["gallery"]
        with self.assertRaisesRegex(ValueError, "Model emitted shape"):
            extract_descriptors(ToyModel(dim=3), self.root, images, DESCRIPTOR_DIM, image_size=(32, 32))

    def test_missing_and_incorrect_checkpoint(self):
        checkpoint = self.root / "weights.pth"
        with self.assertRaisesRegex(FileNotFoundError, "CosPlace checkpoint missing"):
            load_cosplace(checkpoint)
        checkpoint.write_bytes(b"invalid checkpoint")
        with self.assertRaisesRegex(ValueError, "Unexpected SHA-256"):
            load_cosplace(checkpoint)

    def test_offline_cli_smoke(self):
        with patch("vpr_baseline.cli.load_cosplace", return_value=ToyModel()):
            for split in SPLITS:
                self.run_cli("extract", split, "--checkpoint", "unused", "--height", "64", "--width", "64")
        cache = load_cache(self.cache / "train.npz")
        self.assertEqual([row.image_id for row in cache.gallery.images], ["train-blue", "train-red"])
        self.assertEqual([row.image_id for row in cache.query.images], ["train-query"])
        self.assertEqual(cache.gallery.descriptors.shape, (2, DESCRIPTOR_DIM))
        self.assertEqual(cache.query.descriptors.shape, (1, DESCRIPTOR_DIM))
        with patch("vpr_baseline.cli.load_cosplace", side_effect=AssertionError("evaluation loaded a model")):
            report = self.run_cli("evaluate", "val")
        self.assertEqual([report[f"Recall@{k}"] for k in (1, 5, 10)], [1.0, 1.0, 1.0])

        manifest = self.root / "val/gallery.csv"
        header, blue, red = manifest.read_text(encoding="utf-8").splitlines()
        manifest.write_text(f"{header}\n{red}\n{blue}\n", encoding="utf-8")
        errors = io.StringIO()
        with self.assertRaises(SystemExit) as failure, redirect_stderr(errors):
            self.run_cli("evaluate", "val")
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("cached split or image rows differ", errors.getvalue())

    @unittest.skipUnless(os.environ.get("COSPLACE_CHECKPOINT"), "set COSPLACE_CHECKPOINT for real-weight smoke")
    def test_official_checkpoint_cli_smoke(self):
        report = self.run_cli(
            "extract", "val", "--checkpoint", os.environ["COSPLACE_CHECKPOINT"],
            "--height", "64", "--width", "80", "--batch-size", "2",
        )
        self.assertEqual(report["descriptor_dim"], 512)
        cache = load_cache(self.cache / "val.npz")
        for group, count in ((cache.gallery, 2), (cache.query, 1)):
            self.assertEqual(group.descriptors.shape, (count, 512))
            self.assertEqual(group.descriptors.dtype, torch.float32)
            torch.testing.assert_close(group.descriptors.norm(dim=1), torch.ones(count))
        self.assertEqual(self.run_cli("evaluate", "val")["Recall@1"], 1.0)
