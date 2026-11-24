"""Geometry and ranking diagnostics for frozen and adapted descriptors."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .retrieval import l2_normalize

RANK_RTOL = 1e-5


@dataclass(frozen=True)
class DescriptorGeometry:
    cosine_mean: float
    cosine_std: float
    total_variance: float
    numerical_rank: int
    effective_rank: float
    singular_values: Tensor


def descriptor_geometry(descriptors: Tensor) -> DescriptorGeometry:
    """Use unordered distinct-image pairs and population covariance of unit rows."""
    vectors = l2_normalize(descriptors.double())
    if len(vectors) < 2:
        raise ValueError("descriptor geometry needs at least two images")
    first, second = torch.triu_indices(len(vectors), len(vectors), offset=1, device=vectors.device)
    similarities = (vectors @ vectors.T)[first, second].clamp(-1, 1)
    # Subtract a common anchor before the mean so identical rows center exactly to zero.
    shifted = vectors - vectors[:1]
    centered = shifted - shifted.mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    effective_rank = 0.0
    if total > 0:
        fractions = energy[energy > 0] / total
        effective_rank = (-(fractions * fractions.log()).sum()).exp().item()
    return DescriptorGeometry(
        similarities.mean().item(), similarities.std(correction=0).item(),
        (total / len(vectors)).item(), int((singular > singular[0] * RANK_RTOL).sum().item()),
        effective_rank, singular,
    )


def pair_similarities(
    queries: Tensor, gallery: Tensor,
    positives: Sequence[Collection[int]], negatives: Sequence[Collection[int]],
) -> tuple[Tensor, Tensor]:
    """Score known positives and eligible negatives by cosine, preserving their classes."""
    if len(positives) != len(queries) or len(negatives) != len(queries):
        raise ValueError("each query needs a positive and a negative set")
    scores = (l2_normalize(queries.double()) @ l2_normalize(gallery.double()).T).clamp(-1, 1)
    matched, unmatched = [], []
    for i, (positive, negative) in enumerate(zip(positives, negatives, strict=True)):
        if not positive or not negative or set(positive).intersection(negative):
            raise ValueError("positive and negative sets must be nonempty and disjoint")
        if any(index < 0 or index >= len(gallery) for index in (*positive, *negative)):
            raise ValueError("pair labels must refer to the current gallery")
        matched.append(scores[i, sorted(positive)])
        unmatched.append(scores[i, sorted(negative)])
    return torch.cat(matched), torch.cat(unmatched)


def first_positive_ranks(ranking: Tensor, positives: Sequence[Collection[int]]) -> Tensor:
    """Return the one-based rank of each query's highest-ranked labeled positive."""
    if ranking.ndim != 2 or ranking.shape[0] != len(positives):
        raise ValueError("ranking and labels must describe the same queries")
    ranks = []
    for order, matches in zip(ranking.tolist(), positives, strict=True):
        rank = next((place for place, index in enumerate(order, start=1) if index in matches), None)
        if rank is None:
            raise ValueError("each query needs a positive in the full gallery ranking")
        ranks.append(rank)
    return torch.tensor(ranks, dtype=torch.long)


def rank_changes(original: Tensor, adapted: Tensor) -> dict[str, int]:
    if original.ndim != 1 or original.shape != adapted.shape:
        raise ValueError("compare one rank per query on the same split")
    return {
        "rank_improved": int((adapted < original).sum().item()),
        "rank_worsened": int((adapted > original).sum().item()),
        "rank_unchanged": int((adapted == original).sum().item()),
    }


def residual_stats(original: Tensor, adapted: Tensor, delta: Tensor, alpha: float) -> dict[str, float]:
    """Measure alignment, relative correction strength, and sample-specific variation."""
    original, adapted, update = original.double(), adapted.double(), alpha * delta.double()
    cosine = (l2_normalize(original) * l2_normalize(adapted)).sum(dim=1).clamp(-1, 1)
    ratio = update.norm(dim=1) / original.norm(dim=1)
    update_energy = update.square().sum()
    varying_fraction = 0.0
    if update_energy > 0:
        varying_fraction = ((update - update.mean(dim=0)).square().sum() / update_energy).sqrt().item()
    return {
        "alpha": alpha, "cosine_to_original_mean": cosine.mean().item(),
        "cosine_to_original_std": cosine.std(correction=0).item(),
        "relative_correction_mean": ratio.mean().item(), "relative_correction_std": ratio.std(correction=0).item(),
        "correction_centered_rms_fraction": varying_fraction,
    }
