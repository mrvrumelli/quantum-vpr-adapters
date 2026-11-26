"""Exact cosine ranking and closed-set VPR metrics."""

import torch
from torch import Tensor

RECALL_KS = (1, 5, 10)


class ZeroDescriptorError(ValueError):
    """A zero descriptor has no direction for cosine retrieval."""


def l2_normalize(descriptors: Tensor) -> Tensor:
    if descriptors.ndim != 2 or min(descriptors.shape) == 0 or not descriptors.is_floating_point():
        raise ValueError("expected a non-empty floating-point descriptor matrix (N, D)")
    norms = torch.linalg.vector_norm(descriptors, dim=1, keepdim=True)
    if not torch.isfinite(norms).all():
        raise ValueError("descriptors must have finite, nonzero L2 norms")
    if (norms == 0).any():
        raise ZeroDescriptorError("descriptors must have finite, nonzero L2 norms")
    return descriptors / norms


def rank_cosine(queries: Tensor, gallery: Tensor, k: int, query_batch_size: int = 256) -> Tensor:
    """Return gallery-row indices in decreasing cosine order, breaking ties by gallery order."""
    if k <= 0 or query_batch_size <= 0:
        raise ValueError("k and query batch size must be positive")
    queries = l2_normalize(queries)
    gallery = l2_normalize(gallery)
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery descriptor dimensions must match")
    rankings = []
    for batch in queries.split(query_batch_size):
        similarities = batch @ gallery.T
        order = torch.argsort(similarities, dim=1, descending=True, stable=True)
        rankings.append(order[:, :k].clone())
    return torch.cat(rankings)


def recall_at_k(
    ranked_indices: Tensor, positives: list[set[int]], ks: tuple[int, ...] = RECALL_KS
) -> dict[int, float]:
    """Return the fraction of queries with any positive in the first k available ranks."""
    if ranked_indices.ndim != 2 or min(ranked_indices.shape) == 0:
        raise ValueError("expected a non-empty ranking matrix (queries, ranks)")
    if len(positives) != ranked_indices.shape[0] or not all(positives):
        raise ValueError("each ranked query must have a non-empty positive set")
    if any(k <= 0 for k in ks):
        raise ValueError("recall cutoffs must be positive")
    first_hits = [
        next((rank for rank, index in enumerate(row, start=1) if index in matches), float("inf"))
        for row, matches in zip(ranked_indices.tolist(), positives, strict=True)
    ]
    return {k: sum(rank <= k for rank in first_hits) / len(first_hits) for k in ks}


def evaluate_recall(
    queries: Tensor, gallery: Tensor, positives: list[set[int]], query_batch_size: int = 256
) -> dict[str, float]:
    rankings = rank_cosine(queries, gallery, max(RECALL_KS), query_batch_size)
    return {f"Recall@{k}": value for k, value in recall_at_k(rankings, positives).items()}
