import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from vpr_baseline.adapters import AdapterConfig, evaluate_adapter, load_adapter
from vpr_baseline.cache import DescriptorSet, SplitCache, save_cache
from vpr_baseline.cli import main
from vpr_baseline.data import ImageRecord
from vpr_baseline.retrieval import rank_cosine, recall_at_k
from vpr_baseline.training import TrainConfig, build_triplet_pool, train_adapter


def _records(split, role, frames):
    return tuple(ImageRecord(f"{split}-{role}-{frame}", f"{split}/{role}/Image{frame:03}.jpg") for frame in frames)


def _synthetic_cache(split, offset, requires_grad=False):
    vectors = torch.eye(512)[offset:offset + 3]
    groups = {}
    for role in ("gallery", "query"):
        rows = _records(split, role, (offset, offset + 10, offset + 20))
        groups[role] = DescriptorSet(rows, vectors.clone().requires_grad_(requires_grad))
    return SplitCache(split, {"descriptor_dim": 512}, **groups)


def _same_frame_labels(cache):
    return "query_id,gallery_id\n" + "".join(
        f"{query.image_id},{gallery.image_id}\n"
        for query, gallery in zip(cache.query.images, cache.gallery.images, strict=True)
    )


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_triplets_exclude_nearby_frames_even_when_rows_are_out_of_order(self):
        gallery = _records("train", "gallery", (30, 10, 14, 15, 16, 25))
        queries = _records("train", "query", (10,))
        pool = build_triplet_pool(gallery, queries, [{1, 5}], window=5)
        self.assertEqual(pool.negatives, ((4,),))
        self.assertEqual(pool.all_triplets().tolist(), [[0, 1, 4], [0, 5, 4]])
        first = pool.sample(torch.Generator().manual_seed(13))
        second = pool.sample(torch.Generator().manual_seed(13))
        torch.testing.assert_close(first, second)
        self.assertIn(first.tolist()[0], pool.all_triplets().tolist())

    def test_triplets_reject_missing_labels_or_empty_negative_pool(self):
        gallery = _records("train", "gallery", (0, 2, 4))
        queries = _records("train", "query", (0,))
        with self.assertRaisesRegex(ValueError, "missing or invalid positive"):
            build_triplet_pool(gallery, queries, [set()], window=0)
        with self.assertRaisesRegex(ValueError, "no eligible negative"):
            build_triplet_pool(gallery, queries, [{0}], window=4)

    def test_training_keeps_features_frozen_and_validation_out_of_gradients(self):
        train = _synthetic_cache("train", 0, requires_grad=True)
        val = _synthetic_cache("val", 40, requires_grad=True)
        fixed = [{0}, {1}, {2}]
        other = [{1}, {2}, {0}]
        config = TrainConfig(epochs=3, batch_size=2, patience=3, seed=17)
        adapter = AdapterConfig("residual")
        a = train_adapter(adapter, config, train, fixed, val, fixed, self.root / "a")
        b = train_adapter(adapter, config, train, fixed, val, other, self.root / "b")
        self.assertEqual((a["epochs_run"], b["epochs_run"]), (3, 3))
        self.assertEqual([row["train_loss"] for row in a["history"]], [row["train_loss"] for row in b["history"]])
        self.assertNotEqual(a["history"][0]["val_loss"], b["history"][0]["val_loss"])
        for cache, offset in ((train, 0), (val, 40)):
            for role in (cache.gallery, cache.query):
                self.assertIsNone(role.descriptors.grad)
                torch.testing.assert_close(role.descriptors, torch.eye(512)[offset:offset + 3])
        with self.assertRaisesRegex(ValueError, "requires train and val caches"):
            train_adapter(adapter, config, val, fixed, train, fixed, self.root / "invalid")

    def test_early_stopping_retains_first_best_checkpoint(self):
        train = _synthetic_cache("train", 0)
        val = _synthetic_cache("val", 40)
        mapping = [{0}, {1}, {2}]
        output = self.root / "run"
        with patch("vpr_baseline.training._validate", return_value=(0.2, {
            "Recall@1": 0.5, "Recall@5": 1.0, "Recall@10": 1.0,
        })):
            result = train_adapter(
                AdapterConfig("direct"), TrainConfig(epochs=8, patience=2), train, mapping, val, mapping, output
            )
        self.assertEqual((result["best_epoch"], result["epochs_run"]), (1, 3))
        _, epoch = load_adapter(output / "best.pt", val)
        self.assertEqual(epoch, 1)

    def test_cli_training_and_comparison_use_caches_and_phase_one_metrics(self):
        for split, offset in (("train", 0), ("val", 40), ("test", 80)):
            cache = _synthetic_cache(split, offset)
            folder = self.root / split
            folder.mkdir()
            for role in ("gallery", "query"):
                rows = getattr(cache, role).images
                (folder / f"{role}.csv").write_text(
                    "image_id,path\n" + "".join(f"{row.image_id},{row.path}\n" for row in rows), encoding="utf-8"
                )
            if split == "test":
                (folder / "positives.csv").write_text("invalid test labels\n", encoding="utf-8")
                continue
            (folder / "positives.csv").write_text(_same_frame_labels(cache), encoding="utf-8")
            save_cache(self.root / "cache" / f"{split}.npz", cache)

        def run(command, *args):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main([command, "--dataset-root", str(self.root),
                                       "--cache-dir", str(self.root / "cache"), *args]), 0)
            return json.loads(stdout.getvalue())

        mapping = [{0}, {1}, {2}]
        val = _synthetic_cache("val", 40)
        checkpoints = {}
        with patch("vpr_baseline.cli.load_cosplace", side_effect=AssertionError("backbone was loaded")):
            for kind in ("direct", "residual"):
                summary = run("train-adapter", "--adapter", kind, "--epochs", "2", "--output-dir", str(self.root / kind))
                checkpoints[kind] = summary["checkpoint"]
                single = run("evaluate", "--split", "val", "--adapter-checkpoint", checkpoints[kind])
                self.assertEqual(single["Recall@1"], summary["best_validation_recall"]["Recall@1"])
            report_path = self.root / "comparison.json"
            flags = ("--direct-checkpoint", checkpoints["direct"], "--residual-checkpoint", checkpoints["residual"])
            report = run("compare-adapters", "--split", "val", *flags, "--output", str(report_path))

            test = _synthetic_cache("test", 80)
            save_cache(self.root / "cache/test.npz", test)
            (self.root / "test/positives.csv").write_text(_same_frame_labels(test), encoding="utf-8")
            held_out = run("compare-adapters", "--split", "test", *flags)
            self.assertEqual(held_out["split"], "test")
            self.assertEqual(held_out["models"]["cosplace"]["Recall@1"], 1.0)
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)
        baseline = recall_at_k(rank_cosine(val.query.descriptors, val.gallery.descriptors, 10), mapping)
        self.assertEqual(report["models"]["cosplace"], {f"Recall@{k}": value for k, value in baseline.items()})
        for kind, path in checkpoints.items():
            model, _ = load_adapter(Path(path), val)
            with torch.inference_mode():
                ranking = rank_cosine(model(val.query.descriptors), model(val.gallery.descriptors), 10)
            expected = {f"Recall@{k}": value for k, value in recall_at_k(ranking, mapping).items()}
            self.assertEqual(evaluate_adapter(model, val, mapping), expected)
            self.assertEqual({key: report["models"][kind][key] for key in expected}, expected)
