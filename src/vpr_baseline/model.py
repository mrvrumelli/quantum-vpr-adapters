"""Local inference architecture based on gmberton/CosPlace (see LICENSES/CosPlace.txt)."""

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.functional import avg_pool2d, normalize
from torchvision.models import resnet18

DESCRIPTOR_DIM = 512
CHECKPOINT_NAME = "resnet18_512_cosplace.pth"
CHECKPOINT_URL = f"https://github.com/gmberton/CosPlace/releases/download/v1.0/{CHECKPOINT_NAME}"
CHECKPOINT_SHA256 = "db09310a175e325fe25e7a460b41886ab9357b19c41d370a9ed3ed81bc9abf9f"


class _ChannelNormalize(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return normalize(features, dim=1)


class _GeM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), 3.0))

    def forward(self, features: Tensor) -> Tensor:
        powered = features.clamp_min(1e-6).pow(self.p)
        return avg_pool2d(powered, features.shape[-2:]).pow(self.p.reciprocal())


class _CosPlace(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.aggregation = nn.Sequential(
            _ChannelNormalize(),
            _GeM(),
            nn.Flatten(1),
            nn.Linear(resnet.fc.in_features, DESCRIPTOR_DIM),
            _ChannelNormalize(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.aggregation(self.backbone(images))


def load_cosplace(checkpoint: Path) -> nn.Module:
    """Verify the exact official asset and load every model parameter on CPU."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CosPlace checkpoint missing at {checkpoint}. Download {CHECKPOINT_URL}")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise ValueError(f"Unexpected SHA-256 for {checkpoint}: {digest}. Download {CHECKPOINT_URL}")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)

    model = _CosPlace()
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).eval()
