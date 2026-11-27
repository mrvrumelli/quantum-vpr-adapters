"""Record bottleneck mixer and projection gradients during adapter training."""

from statistics import mean

import torch
from torch import Tensor, nn

from .adapters import MLPAdapter


def _gradients(module: nn.Module) -> Tensor | None:
    grads = [p.grad.detach() if p.grad is not None else torch.zeros_like(p)
             for p in module.parameters() if p.requires_grad]
    return torch.cat([grad.flatten() for grad in grads]) if grads else None


class MixerMonitor:
    def __init__(self, model: MLPAdapter) -> None:
        if model.mixer is None:
            raise ValueError("mixer diagnostics require a bottleneck mixer")
        self.model = model
        self.mixer = model.mixer
        self.initial = torch.cat([p.detach().flatten() for p in self.mixer.parameters()])
        self.samples: list[dict] = []

    @torch.no_grad()
    def observe(self) -> None:
        mixer = _gradients(self.mixer)
        down = _gradients(self.model.down)
        bottleneck = _gradients(self.model.mlp)
        up = _gradients(self.model.up)
        if any(not torch.isfinite(grad).all() for grad in (mixer, down, bottleneck, up) if grad is not None):
            raise ValueError("nonfinite adapter gradient")
        self.samples.append({
            "mixer_gradient_norm": mixer.norm().item() if mixer is not None else None,
            "mixer_gradient_variance": mixer.var(correction=0).item() if mixer is not None else None,
            "input_projection_gradient_norm": torch.cat((down, bottleneck)).norm().item(),
        })

    @torch.no_grad()
    def finish_epoch(self, seconds: float) -> dict:
        metrics = {}
        for key in self.samples[0]:
            values = [sample[key] for sample in self.samples if sample[key] is not None]
            metrics[key] = mean(values) if values else None
        self.samples.clear()
        parameters = torch.cat([p.detach().flatten() for p in self.mixer.parameters()])
        return {
            **metrics, "epoch_seconds": seconds, "mixer_parameters": parameters.tolist(),
            "mixer_parameter_drift": (parameters - self.initial).norm().item(),
            "residual_scale": self.model.config.alpha if self.model.config.kind == "residual" else None,
        }
