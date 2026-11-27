"""Matched residual-adapter diagnostics and route-block uncertainty estimates."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .adapters import MLPAdapter
from .cache import SplitCache
from .diagnostics import descriptor_geometry
from .retrieval import RECALL_KS, first_positive_ranks, l2_normalize, rank_cosine

RETRIEVAL_METRICS = ("MRR", *(f"Recall@{k}" for k in RECALL_KS))


@dataclass(frozen=True)
class ResidualSnapshot:
    metrics: dict[str, float]
    ranks: Tensor
    scores: Tensor
    descriptor_norms: Tensor
    unscaled_correction: Tensor
    scaled_correction: Tensor
    descriptor_cosines: Tensor
    covariance: Tensor | None


def distribution(values: Tensor, label: str) -> dict[str, float]:
    values = values.double().flatten()
    quantiles = values.quantile(values.new_tensor([0.05, 0.50, 0.95])).tolist()
    return {f"{label}_{stat}": value for stat, value in zip(
        ("mean", "sd", "p05", "median", "p95"),
        (values.mean().item(), values.std(correction=0).item(), *quantiles), strict=True,
    )}


def feature_covariance(features: Tensor) -> tuple[Tensor, float]:
    """Population covariance and entropy rank of centered, unnormalized readouts."""
    shifted = features.double() - features[:1].double()
    centered = shifted - shifted.mean(dim=0)
    covariance = centered.T @ centered / len(centered)
    energy = torch.linalg.eigvalsh(covariance).clamp_min(0)
    if energy.sum() == 0:
        return covariance, 0.0
    fractions = energy[energy > 0] / energy.sum()
    return covariance, (-(fractions * fractions.log()).sum()).exp().item()


def retrieval_values(ranks: Tensor) -> dict[str, Tensor]:
    """Keep the paired per-query reciprocal ranks and hit indicators."""
    return {"MRR": ranks.double().reciprocal(), **{f"Recall@{k}": (ranks <= k).double() for k in RECALL_KS}}


@torch.inference_mode()
def measure_snapshot(cache: SplitCache, positives: list[set[int]], model: MLPAdapter | None) -> ResidualSnapshot:
    queries, gallery = cache.query.descriptors, cache.gallery.descriptors
    original = torch.cat((queries, gallery))
    covariance = None
    metrics = {}
    unscaled = scaled = torch.zeros(len(original), dtype=torch.float64)
    adapted = original
    if model is not None:
        query, reference = (model.components(group) for group in (queries, gallery))
        readout = torch.cat((query[1], reference[1]))
        delta = torch.cat((query[2], reference[2])).double()
        unscaled = delta.norm(dim=1)
        scaled = model.config.alpha * unscaled
        queries, gallery = (l2_normalize(raw) for raw in (query[3], reference[3]))
        adapted = torch.cat((queries, gallery))
        covariance, rank = feature_covariance(readout)
        metrics = {"feature_effective_rank": rank, "feature_total_variance": covariance.trace().item(),
                   **{f"feature_{i}_variance": value for i, value in enumerate(covariance.diagonal().tolist())}}

    ranking = rank_cosine(queries, gallery, len(gallery))
    ranks = first_positive_ranks(ranking, positives)
    scores = (l2_normalize(queries.double()) @ l2_normalize(gallery.double()).T).clamp(-1, 1)
    geometry = descriptor_geometry(adapted)
    norms = original.double().norm(dim=1)
    cosines = (l2_normalize(original.double()) * l2_normalize(adapted.double())).sum(1).clamp(-1, 1)
    metrics.update({
        **{metric: values.mean().item() for metric, values in retrieval_values(ranks).items()},
        "effective_rank": geometry.effective_rank, "descriptor_variance": geometry.total_variance,
        "distinct_cosine_mean": geometry.cosine_mean,
        **distribution(norms, "original_norm"), **distribution(unscaled, "unscaled_correction"),
        **distribution(scaled, "scaled_correction"), **distribution(scaled / norms, "relative_correction"),
        **distribution(cosines, "cosine_to_original"),
    })
    return ResidualSnapshot(metrics, ranks, scores, norms, unscaled, scaled, cosines, covariance)


def paired_block_bootstrap(
    differences: np.ndarray, frames: tuple[int, ...], *, block_length: int = 5, repeats: int = 20000,
    seed: int = 3202,
) -> tuple[float, float]:
    """Resample matched seed rows and contiguous frame blocks jointly across seeds."""
    if differences.ndim != 2 or differences.shape[1] != len(frames) or not differences.shape[0]:
        raise ValueError("expected one column per query and one row per matched seed")
    if block_length < 1 or repeats < 1 or len(frames) % block_length or len(frames) // block_length < 2:
        raise ValueError("use at least two equal-length route blocks")
    order = np.argsort(frames)
    if not np.all(np.diff(np.asarray(frames)[order]) == 1):
        raise ValueError("the route blocks require unique consecutive query frames")
    if not np.isfinite(differences).all():
        raise ValueError("paired effects must be finite")
    seeds = len(differences)
    blocks = len(frames) // block_length
    block_means = differences[:, order].reshape(seeds, blocks, block_length).mean(axis=2)
    rng = np.random.default_rng(seed)
    sampled_seeds = rng.integers(0, seeds, size=(repeats, seeds, 1))
    sampled_blocks = rng.integers(0, blocks, size=(repeats, 1, blocks))
    bootstrap_means = block_means[sampled_seeds, sampled_blocks].mean(axis=(1, 2))
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)
