import unittest

import torch

from vpr_baseline.retrieval import l2_normalize, rank_cosine, recall_at_k


class RetrievalTests(unittest.TestCase):
    def test_normalization(self):
        vectors = torch.tensor([[3.0, 4.0], [0.0, -2.0]])
        torch.testing.assert_close(l2_normalize(vectors), torch.tensor([[0.6, 0.8], [0.0, -1.0]]))

    def test_normalization_rejects_zero_and_nonfinite_rows(self):
        for row in ([[0.0, 0.0]], [[float("nan"), 1.0]], [[1.0, float("inf")]]):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "finite, nonzero"):
                l2_normalize(torch.tensor(row))

    def test_cosine_ranking_is_scale_invariant_and_stable_on_ties(self):
        gallery = torch.tensor([[6.0, 8.0], [1.0, 0.0], [4.0, 0.0], [-4.0, 0.0]])
        queries = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
        expected = torch.tensor([[1, 2, 0, 3], [3, 0, 1, 2]])
        torch.testing.assert_close(rank_cosine(queries, gallery, k=10, query_batch_size=1), expected)

    def test_recall_at_1_5_10_with_multiple_positives_and_a_miss(self):
        rankings = torch.arange(12).repeat(4, 1)
        positives = [{0, 11}, {4}, {9}, {10, 11}]
        self.assertEqual(recall_at_k(rankings, positives), {1: 0.25, 5: 0.5, 10: 0.75})

    def test_recall_with_small_gallery(self):
        self.assertEqual(recall_at_k(torch.tensor([[0, 1]]), [{1}]), {1: 0.0, 5: 1.0, 10: 1.0})
