"""Train descriptor adapters with split-local, temporally filtered triplets."""

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch import Tensor

from .adapters import CHECKPOINT_VERSION, AdapterConfig, MLPAdapter, feature_spec
from .cache import SplitCache
from .data import ImageRecord
from .mixer_monitor import MixerMonitor
from .retrieval import RECALL_KS, ZeroDescriptorError, evaluate_recall


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    learning_rate: float = 0.001
    margin: float = 0.2
    batch_size: int = 16
    negative_window: int = 5
    patience: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.patience) <= 0:
            raise ValueError("epochs, batch size and patience must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(self.margin) or self.margin <= 0:
            raise ValueError("margin must be finite and positive")
        if self.negative_window < 0 or not 0 <= self.seed < 2**63:
            raise ValueError("negative window and seed must be nonnegative, and seed must be below 2**63")


@dataclass(frozen=True)
class TripletPool:
    positives: tuple[tuple[int, ...], ...]
    negatives: tuple[tuple[int, ...], ...]

    def sample(self, generator: torch.Generator) -> Tensor:
        """Draw one positive and one eligible negative per query in shuffled order."""
        triplets = []
        for query in torch.randperm(len(self.positives), generator=generator).tolist():
            positives, negatives = self.positives[query], self.negatives[query]
            positive = positives[torch.randint(len(positives), (), generator=generator).item()]
            negative = negatives[torch.randint(len(negatives), (), generator=generator).item()]
            triplets.append((query, positive, negative))
        return torch.tensor(triplets)

    def all_triplets(self) -> Tensor:
        return torch.tensor([
            (query, positive, negative)
            for query, (positives, negatives) in enumerate(zip(self.positives, self.negatives, strict=True))
            for positive in positives
            for negative in negatives
        ])


def build_triplet_pool(
    gallery: tuple[ImageRecord, ...], queries: tuple[ImageRecord, ...],
    positives: list[set[int]], window: int,
) -> TripletPool:
    if window < 0 or not queries or len(positives) != len(queries):
        raise ValueError("triplets require a nonnegative window and positive labels for every query")
    frames = []
    for record in gallery:
        match = re.fullmatch(r"(?:Image)?([0-9]+)", Path(record.path).stem, re.IGNORECASE)
        if match is None:
            raise ValueError(f"cannot infer a gallery frame number from {record.path}")
        frames.append(int(match[1]))

    positive_rows = []
    negative_rows = []
    for record, matches in zip(queries, positives, strict=True):
        if not matches or any(index < 0 or index >= len(gallery) for index in matches):
            raise ValueError(f"missing or invalid positive for query {record.image_id}")
        positive_frames = {frames[index] for index in matches}
        eligible = tuple(
            index for index, frame in enumerate(frames)
            if all(abs(frame - positive) > window for positive in positive_frames)
        )
        if not eligible:
            raise ValueError(f"no eligible negative for query {record.image_id} with window {window}")
        positive_rows.append(tuple(sorted(matches)))
        negative_rows.append(eligible)
    return TripletPool(tuple(positive_rows), tuple(negative_rows))


def _hinge(positive: Tensor, negative: Tensor, margin: float) -> Tensor:
    return (margin + negative - positive).clamp_min(0).mean()


@torch.inference_mode()
def _validate(
    model: MLPAdapter, cache: SplitCache, positives: list[set[int]], triplets: Tensor, margin: float
) -> tuple[float, dict[str, float]]:
    model.eval()
    queries = model(cache.query.descriptors)
    gallery = model(cache.gallery.descriptors)
    scores = queries @ gallery.T
    q, p, n = triplets.unbind(dim=1)
    loss = _hinge(scores[q, p], scores[q, n], margin).item()
    return loss, evaluate_recall(queries, gallery, positives)


def train_adapter(
    adapter: AdapterConfig, config: TrainConfig,
    train: SplitCache, train_positives: list[set[int]],
    val: SplitCache, val_positives: list[set[int]], output_dir: Path,
) -> dict:
    if train.split != "train" or val.split != "val":
        raise ValueError("adapter training requires train and val caches in that order")
    if output_dir.exists():
        raise FileExistsError(f"training output already exists: {output_dir}")
    features = feature_spec(train)
    if features != feature_spec(val):
        raise ValueError("train and validation descriptor caches use different features")
    train_pool = build_triplet_pool(train.gallery.images, train.query.images, train_positives, config.negative_window)
    val_pool = build_triplet_pool(val.gallery.images, val.query.images, val_positives, config.negative_window)
    val_triplets = val_pool.all_triplets()

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        model = MLPAdapter(adapter)
    monitor = MixerMonitor(model) if model.mixer is not None else None
    rng = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    train_queries = train.query.descriptors.detach()
    train_gallery = train.gallery.descriptors.detach()
    history = []
    best_score = ()
    best_epoch = 0
    best_state = {}
    for epoch in range(1, config.epochs + 1):
        if monitor is not None:
            started = perf_counter()
        model.train()
        total_loss = 0.0
        triplets = train_pool.sample(rng)
        for batch in triplets.split(config.batch_size):
            q, p, n = batch.unbind(dim=1)
            optimizer.zero_grad(set_to_none=True)
            try:
                queries = model(train_queries[q])
                positive, negative = model(train_gallery[torch.cat((p, n))]).chunk(2)
            except ZeroDescriptorError as error:
                error.epoch, error.split = epoch, "train"
                raise
            loss = _hinge((queries * positive).sum(dim=1), (queries * negative).sum(dim=1), config.margin)
            loss.backward()
            if monitor is not None:
                monitor.observe()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        try:
            val_loss, val_recall = _validate(model, val, val_positives, val_triplets, config.margin)
        except ZeroDescriptorError as error:
            error.epoch, error.split = epoch, "val"
            raise
        entry = {
            "epoch": epoch, "train_loss": total_loss / len(triplets),
            "val_loss": val_loss, "val_recall": val_recall,
        }
        if monitor is not None:
            entry["mixer_diagnostics"] = monitor.finish_epoch(perf_counter() - started)
        history.append(entry)
        score = tuple(val_recall[f"Recall@{k}"] for k in RECALL_KS)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch - best_epoch >= config.patience:
            break

    checkpoint = {
        "version": CHECKPOINT_VERSION, "adapter": asdict(adapter), "feature_spec": features,
        "epoch": best_epoch, "state_dict": best_state,
    }
    report = {
        "adapter": asdict(adapter), "training": asdict(config), "feature_spec": features,
        "torch": str(torch.__version__), "device": "cpu", "threads": torch.get_num_threads(),
        "train_query_count": len(train_queries), "val_query_count": len(val.query.images),
        "validation_triplet_count": len(val_triplets), "epochs_run": len(history),
        "best_epoch": best_epoch, "best_validation_recall": history[best_epoch - 1]["val_recall"],
        "history": history,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        torch.save(checkpoint, staging / "best.pt")
        (staging / "history.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    return report
