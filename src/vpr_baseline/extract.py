"""Batched PyTorch descriptor extraction in manifest order."""

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from .data import ImageRecord
from .retrieval import l2_normalize

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = (384, 512)


class _Images(Dataset):
    def __init__(self, root: Path, records: tuple[ImageRecord, ...], image_size: tuple[int, int]):
        self.root = root
        self.records = records
        self.transform = Compose([
            Resize(image_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            ToTensor(),
            Normalize(IMAGE_MEAN, IMAGE_STD),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tensor:
        path = self.root / self.records[index].path
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


@torch.inference_mode()
def extract_descriptors(
    model: nn.Module,
    root: Path,
    records: tuple[ImageRecord, ...],
    descriptor_dim: int,
    *,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 8,
    workers: int = 0,
    device: str = "cpu",
) -> Tensor:
    if not records or min(image_size) <= 0 or batch_size <= 0 or workers < 0:
        raise ValueError("expected images, positive image/batch sizes and nonnegative workers")
    target = torch.device(device)
    model.requires_grad_(False).eval().to(target)
    loader = DataLoader(
        _Images(root, records, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=target.type == "cuda",
    )
    batches = []
    for images in loader:
        descriptors = model(images.to(target))
        expected = (len(images), descriptor_dim)
        if descriptors.shape != expected:
            raise ValueError(f"Model emitted shape {tuple(descriptors.shape)}, expected {expected}")
        batches.append(l2_normalize(descriptors.to(device="cpu", dtype=torch.float32)))
    return torch.cat(batches)
