"""Retrieval and representation diagnostics for the controlled mixer comparison."""

import math

import torch

from .adapters import MLPAdapter
from .cache import SplitCache
from .diagnostics import descriptor_geometry, pair_similarities, residual_stats
from .quantum import encode_angles
from .retrieval import evaluate_retrieval, l2_normalize

NEAR_ZERO = 1e-6


@torch.inference_mode()
def evaluate_descriptors(
    cache: SplitCache, positives: list[set[int]], negatives: tuple[tuple[int, ...], ...],
    model: MLPAdapter | None,
) -> dict:
    inputs = (cache.query.descriptors, cache.gallery.descriptors)
    raw = inputs
    metrics = {}
    if model is not None:
        query, gallery = (model.components(group) for group in inputs)
        raw = (query[3], gallery[3])
        if model.mixer is not None:
            features = torch.cat((query[1], gallery[1])).double()
            encoded = encode_angles(torch.cat((query[0], gallery[0])))
            metrics["angle_saturation_fraction"] = (encoded.abs() >= 0.95 * math.pi).double().mean().item()
            for wire, values in enumerate(features.T):
                metrics[f"mixer_output_{wire}_mean"] = values.mean().item()
                metrics[f"mixer_output_{wire}_std"] = values.std(correction=0).item()
    stacked = torch.cat(raw)
    norms = torch.linalg.vector_norm(stacked, dim=1)
    finite = norms[torch.isfinite(norms)].double()
    metrics.update({
        "descriptor_count": len(stacked), "nan_elements": int(stacked.isnan().sum().item()),
        "inf_elements": int(stacked.isinf().sum().item()), "nonfinite_norm_rows": len(norms) - len(finite),
        "zero_norm_rows": int((norms == 0).sum().item()),
        "near_zero_norm_rows": int(((norms > 0) & (norms < NEAR_ZERO)).sum().item()),
    })
    if len(finite):
        metrics.update({"raw_norm_min": finite.min().item(), "raw_norm_mean": finite.mean().item(),
                        "raw_norm_max": finite.max().item(), "raw_norm_std": finite.std(correction=0).item()})
    if metrics["nonfinite_norm_rows"] or metrics["zero_norm_rows"]:
        return {"evaluation_status": "invalid_descriptor", **metrics}

    q, g = (l2_normalize(group) for group in raw) if model is not None else raw
    adapted = torch.cat((q, g))
    geometry = descriptor_geometry(adapted)
    positive, negative = pair_similarities(q, g, positives, negatives)
    metrics.update({"evaluation_status": "evaluated", **evaluate_retrieval(q, g, positives),
                    "pair_cosine_mean": geometry.cosine_mean, "total_variance": geometry.total_variance,
                    "effective_rank": geometry.effective_rank, "positive_cosine_mean": positive.mean().item(),
                    "negative_cosine_mean": negative.mean().item(),
                    "positive_negative_gap": positive.mean().item() - negative.mean().item(),
                    "max_unit_norm_error": (adapted.double().norm(dim=1) - 1).abs().max().item()})
    if model is not None and model.config.kind == "residual":
        metrics.update(residual_stats(torch.cat(inputs), adapted, torch.cat((query[2], gallery[2])), model.config.alpha))
    return metrics
