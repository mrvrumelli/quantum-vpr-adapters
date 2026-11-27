"""Direct and residual adapters for frozen CosPlace descriptors."""

import math
from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError

import torch
from torch import Tensor, nn

from .cache import SplitCache
from .model import DESCRIPTOR_DIM
from .quantum import QUBITS, ClassicalMixer, QuantumMixer
from .retrieval import evaluate_recall, evaluate_retrieval, l2_normalize

ADAPTER_KINDS = ("direct", "residual")
BOTTLENECK_ACTIVATIONS = ("relu", "leaky_relu")
ADAPTER_TRANSFORMS = ("mlp", "quantum", "surrogate", "identity")
LEAKY_RELU_SLOPE = 0.01
RESIDUAL_UP_INIT_SCALE = 0.05
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
    activation: str = "relu"
    transform: str = "mlp"
    circuit_depth: int = 2
    entangle: bool = True
    train_quantum: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ADAPTER_KINDS:
            raise ValueError(f"unknown adapter kind: {self.kind}")
        if any(type(size) is not int or size <= 0 for size in (self.down_dim, self.hidden_dim)):
            raise ValueError("adapter widths must be positive integers")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if type(self.up_bias) is not bool:
            raise ValueError("up_bias must be boolean")
        if self.transform not in ADAPTER_TRANSFORMS:
            raise ValueError(f"unknown bottleneck transform: {self.transform}")
        if self.transform != "mlp" and self.hidden_dim != QUBITS:
            raise ValueError("mixer controls require a four-dimensional bottleneck")
        if type(self.circuit_depth) is not int or self.circuit_depth < 1:
            raise ValueError("circuit depth must be a positive integer")
        if type(self.entangle) is not bool or type(self.train_quantum) is not bool:
            raise ValueError("entangle and train_quantum must be boolean")
        if self.transform != "quantum" and (self.circuit_depth != 2 or not self.entangle or not self.train_quantum):
            raise ValueError("circuit options require the quantum transform")
        classical_residual = self.kind == "residual" and self.transform == "mlp"
        if classical_residual and not self.up_bias:
            raise ValueError("up_bias must be boolean and may only be disabled for the Direct adapter")
        if self.activation not in BOTTLENECK_ACTIVATIONS:
            raise ValueError(f"unknown bottleneck activation: {self.activation}")
        if classical_residual and self.activation != "relu":
            raise ValueError("LeakyReLU is available only for the Direct adapter")


class MLPAdapter(nn.Module):
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.down = nn.Linear(DESCRIPTOR_DIM, config.down_dim)
        activation = nn.LeakyReLU(LEAKY_RELU_SLOPE) if config.activation == "leaky_relu" else nn.ReLU()
        self.mlp = nn.Sequential(nn.Linear(config.down_dim, config.hidden_dim), activation)
        self.up = nn.Linear(config.hidden_dim, DESCRIPTOR_DIM, bias=config.up_bias)
        self.mixer: nn.Module | None = None
        if config.transform == "quantum":
            self.mixer = QuantumMixer(config.circuit_depth, entangle=config.entangle, trainable=config.train_quantum)
        elif config.transform == "surrogate":
            self.mixer = ClassicalMixer()
        elif config.transform == "identity":
            self.mixer = nn.Identity()
        if self.mixer is not None and config.kind == "residual":
            with torch.no_grad():
                self.up.weight.mul_(RESIDUAL_UP_INIT_SCALE)
                if self.up.bias is not None:
                    self.up.bias.mul_(RESIDUAL_UP_INIT_SCALE)

    def components(self, descriptors: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if descriptors.ndim != 2 or descriptors.shape[1] != DESCRIPTOR_DIM:
            raise ValueError(f"expected a (batch, {DESCRIPTOR_DIM}) descriptor matrix")
        bottleneck = self.mlp(self.down(descriptors))
        features = self.mixer(bottleneck) if self.mixer is not None else bottleneck
        delta = self.up(features)
        raw = descriptors + self.config.alpha * delta if self.config.kind == "residual" else delta
        return bottleneck, features, delta, raw

    def forward(self, descriptors: Tensor) -> Tensor:
        return l2_normalize(self.components(descriptors)[-1])


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
    model: MLPAdapter, cache: SplitCache, positives: list[set[int]], query_batch_size: int = 256,
    *, include_mrr: bool = False,
) -> dict[str, float]:
    evaluate = evaluate_retrieval if include_mrr else evaluate_recall
    return evaluate(
        model(cache.query.descriptors), model(cache.gallery.descriptors), positives, query_batch_size
    )
