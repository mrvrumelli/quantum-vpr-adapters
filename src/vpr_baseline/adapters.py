"""Direct and residual MLPs for frozen CosPlace descriptors."""

import math
from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError

import torch
from torch import Tensor, nn

from .cache import SplitCache
from .model import DESCRIPTOR_DIM
from .retrieval import evaluate_recall, l2_normalize

ADAPTER_KINDS = ("direct", "residual")
CHECKPOINT_VERSION = 1
FEATURE_FIELDS = (
    "descriptor_dim", "checkpoint_sha256", "image_size_hw", "resize", "rgb_mean", "rgb_std", "l2_normalized"
)


@dataclass(frozen=True)
class AdapterConfig:
    kind: str
    down_dim: int = 8
    hidden_dim: int = 4
    alpha: float = 0.1
    up_bias: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ADAPTER_KINDS:
            raise ValueError(f"unknown adapter kind: {self.kind}")
        if any(type(size) is not int or size <= 0 for size in (self.down_dim, self.hidden_dim)):
            raise ValueError("adapter widths must be positive integers")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if type(self.up_bias) is not bool or (self.kind == "residual" and not self.up_bias):
            raise ValueError("up_bias must be boolean and may only be disabled for the Direct adapter")


class MLPAdapter(nn.Module):
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.down = nn.Linear(DESCRIPTOR_DIM, config.down_dim)
        self.mlp = nn.Sequential(nn.Linear(config.down_dim, config.hidden_dim), nn.ReLU())
        self.up = nn.Linear(config.hidden_dim, DESCRIPTOR_DIM, bias=config.up_bias)

    def forward(self, descriptors: Tensor) -> Tensor:
        if descriptors.ndim != 2 or descriptors.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(f"expected a (batch, {DESCRIPTOR_DIM}) descriptor matrix")
        delta = self.up(self.mlp(self.down(descriptors)))
        if self.config.kind == "residual":
            delta = descriptors + self.config.alpha * delta
        return l2_normalize(delta)


def feature_spec(cache: SplitCache) -> dict:
    if cache.provenance["descriptor_dim"] != DESCRIPTOR_DIM:
        raise ValueError(f"adapters require {DESCRIPTOR_DIM}-dimensional CosPlace descriptors")
    return {field: cache.provenance[field] for field in FEATURE_FIELDS if field in cache.provenance}


def load_adapter(path: Path, cache: SplitCache) -> tuple[MLPAdapter, int]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["version"] != CHECKPOINT_VERSION:
            raise ValueError(f"{path}: unsupported adapter checkpoint version")
        if checkpoint["feature_spec"] != feature_spec(cache):
            raise ValueError(f"{path}: adapter checkpoint and descriptor cache use different features")
        with torch.random.fork_rng(devices=[]):
            model = MLPAdapter(AdapterConfig(**checkpoint["adapter"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        epoch = checkpoint["epoch"]
        if type(epoch) is not int or epoch < 1:
            raise ValueError(f"{path}: invalid adapter checkpoint epoch")
    except (KeyError, TypeError, EOFError, UnpicklingError) as exc:
        raise ValueError(f"{path}: invalid adapter checkpoint") from exc
    return model.eval(), epoch


@torch.inference_mode()
def evaluate_adapter(
    model: MLPAdapter, cache: SplitCache, positives: list[set[int]], query_batch_size: int = 256
) -> dict[str, float]:
    return evaluate_recall(
        model(cache.query.descriptors), model(cache.gallery.descriptors), positives, query_batch_size
    )
