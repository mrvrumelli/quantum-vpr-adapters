import math
import unittest

import torch

from vpr_baseline.diagnostics import descriptor_geometry, first_positive_ranks, pair_similarities, rank_changes, residual_stats
from vpr_baseline.retrieval import l2_normalize


class DiagnosticsTests(unittest.TestCase):
    def test_identical_vectors_have_no_variance_or_effective_rank(self):
        result = descriptor_geometry(torch.tensor([[1.0, 2.0, 3.0]]).repeat(5, 1))
        self.assertAlmostEqual(result.cosine_mean, 1.0)
        self.assertAlmostEqual(result.cosine_std, 0.0)
        self.assertEqual(result.total_variance, 0.0)
        self.assertEqual(result.numerical_rank, 0)
        self.assertEqual(result.effective_rank, 0.0)

    def test_distinct_pairs_and_balanced_two_dimensional_spectrum(self):
        result = descriptor_geometry(torch.cat((torch.eye(2), -torch.eye(2))))
        self.assertAlmostEqual(result.cosine_mean, -1 / 3)
        self.assertAlmostEqual(result.cosine_std, math.sqrt(2) / 3)
        self.assertAlmostEqual(result.total_variance, 1)
        self.assertEqual(result.numerical_rank, 2)
        self.assertAlmostEqual(result.effective_rank, 2)
        torch.testing.assert_close(result.singular_values, torch.full((2,), math.sqrt(2), dtype=torch.float64))

    def test_similarity_distributions_use_only_the_supplied_pair_sets(self):
        queries = torch.eye(2)
        gallery = torch.cat((queries, -queries[:1]))
        positive, negative = pair_similarities(queries, gallery, [{0}, {1}], [{2}, {0, 2}])
        torch.testing.assert_close(positive, torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(negative, torch.tensor([-1.0, 0, 0], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            pair_similarities(queries, gallery, [{0}, {1}], [{0}, {0}])

    def test_first_positive_rank_and_per_query_changes(self):
        positives = [{0, 2}, {1}, {2}]
        original = first_positive_ranks(torch.tensor([[1, 2, 0], [1, 0, 2], [0, 2, 1]]), positives)
        adapted = first_positive_ranks(torch.tensor([[2, 1, 0], [0, 1, 2], [1, 2, 0]]), positives)
        self.assertEqual(original.tolist(), [2, 1, 2])
        self.assertEqual(rank_changes(original, adapted), {"rank_improved": 1, "rank_worsened": 1, "rank_unchanged": 1})
        with self.assertRaisesRegex(ValueError, "positive in the full gallery"):
            first_positive_ranks(torch.tensor([[1], [1], [2]]), positives)

    def test_residual_norms_use_actual_alpha_and_original_norm(self):
        source = torch.tensor([[2.0, 0], [0, 4.0]])
        delta = torch.tensor([[0.0, 2], [8, 0]])
        alpha = 0.5
        stats = residual_stats(source, l2_normalize(source + alpha * delta), delta, alpha)
        self.assertEqual(stats["alpha"], 0.5)
        self.assertAlmostEqual(stats["relative_correction_mean"], 0.75)
        self.assertAlmostEqual(stats["relative_correction_std"], 0.25)
        self.assertAlmostEqual(stats["cosine_to_original_mean"], (1 / math.sqrt(1.25) + 1 / math.sqrt(2)) / 2)
        self.assertGreater(stats["correction_centered_rms_fraction"], 0)
        zero = residual_stats(source, source, delta, 0)
        self.assertEqual(zero["relative_correction_mean"], 0.0)
        self.assertEqual(zero["correction_centered_rms_fraction"], 0.0)
        self.assertEqual(zero["cosine_to_original_mean"], 1.0)
