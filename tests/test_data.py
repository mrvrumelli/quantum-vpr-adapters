import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vpr_baseline.data import SPLITS, load_manifests, load_positives
from vpr_baseline.gardens import prepare_gardens_point


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for split in SPLITS:
            (self.root / split).mkdir()
            for role in ("gallery", "query"):
                (self.root / split / f"{role}.csv").write_text(
                    f"image_id,path\n{split}-{role},images/{split}-{role}.jpg\n", encoding="utf-8"
                )

    def test_rejects_duplicate_image_path_across_splits(self):
        (self.root / "test/query.csv").write_text(
            "image_id,path\ntest-query,images/train-gallery.jpg\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "reused image ID or path"):
            load_manifests(self.root)

    def test_rejects_outside_paths(self):
        (self.root / "test/query.csv").write_text(
            "image_id,path\ntest-query,../outside.jpg\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "relative to the dataset root"):
            load_manifests(self.root)

    def test_rejects_cross_split_or_role_swapped_labels(self):
        records = load_manifests(self.root)["val"]
        for row in ("val-query,test-gallery", "val-gallery,val-query"):
            with self.subTest(row=row):
                (self.root / "val/positives.csv").write_text(f"query_id,gallery_id\n{row}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unknown query or gallery ID"):
                    load_positives(self.root, "val", records["gallery"], records["query"])

    def test_rejects_queries_without_a_positive(self):
        records = load_manifests(self.root)["val"]
        (self.root / "val/positives.csv").write_text("query_id,gallery_id\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing positive"):
            load_positives(self.root, "val", records["gallery"], records["query"])

    def test_gardens_point_manifest_protocol(self):
        dataset = self.root / "gardens"
        for traversal in ("day_right", "night_right"):
            (dataset / traversal).mkdir(parents=True)
            for frame in range(200):
                (dataset / traversal / f"Image{frame:03d}.jpg").touch()
        prepare_gardens_point(dataset)
        prepare_gardens_point(dataset)
        manifests = load_manifests(dataset)
        self.assertEqual([len(manifests[split]["gallery"]) for split in SPLITS], [80, 50, 50])
        self.assertEqual(manifests["val"]["gallery"][0].image_id, "day_right-090")
        self.assertEqual(manifests["test"]["query"][-1].image_id, "night_right-199")
        self.assertFalse((dataset / "train/positives.csv").exists())
        val = manifests["val"]
        self.assertEqual(load_positives(dataset, "val", val["gallery"], val["query"]), [{i} for i in range(50)])
