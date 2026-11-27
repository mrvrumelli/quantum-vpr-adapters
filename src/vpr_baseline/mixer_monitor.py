"""Record bottleneck mixer and projection gradients during adapter training."""

from math import isfinite
from statistics import mean

import torch
from torch import Tensor, nn

from .adapters import MLPAdapter
from .quantum import QuantumMixer


def _gradients(module: nn.Module) -> Tensor | None:
    grads = [p.grad.detach() if p.grad is not None else torch.zeros_like(p)
             for p in module.parameters() if p.requires_grad]
    return torch.cat([grad.flatten() for grad in grads]) if grads else None


class MixerMonitor:
    def __init__(self, model: MLPAdapter, *, detailed: bool = False) -> None:
        if model.mixer is None:
            raise ValueError("mixer diagnostics require a bottleneck mixer")
        self.model = model
        self.mixer = model.mixer
        self.initial = self._parameters()
        self.samples: list[dict] = []
        self.detailed = detailed
        self.input_gradient_energies: list[float] = []
        if detailed:
            self.mixer.register_forward_pre_hook(self._track_input)

    def _parameters(self) -> Tensor:
        weights = [p.detach().flatten() for p in self.mixer.parameters()]
        return torch.cat(weights) if weights else self.model.up.weight.new_empty(0)

    def _track_input(self, _module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        (features,) = inputs
        if features.requires_grad:
            features.register_hook(self._record_input_gradient)

    def _record_input_gradient(self, grad: Tensor) -> None:
        energy = grad.detach().square().sum().item()
        self.input_gradient_energies.append(energy)

    @torch.no_grad()
    def observe(self) -> None:
        mixer = _gradients(self.mixer)
        down = _gradients(self.model.down)
        bottleneck = _gradients(self.model.mlp)
        up = _gradients(self.model.up)
        if any(not torch.isfinite(grad).all() for grad in (mixer, down, bottleneck, up) if grad is not None):
            raise ValueError("nonfinite adapter gradient")
        sample = {
            "mixer_gradient_norm": mixer.norm().item() if mixer is not None else None,
            "mixer_gradient_variance": mixer.var(correction=0).item() if mixer is not None else None,
            "input_projection_gradient_norm": torch.cat((down, bottleneck)).norm().item(),
        }
        if self.detailed:
            if not self.input_gradient_energies:
                raise RuntimeError("mixer input gradients were not captured")
            energy = sum(self.input_gradient_energies)
            if not isfinite(energy):
                raise ValueError("nonfinite mixer input gradient")
            sample["mixer_input_gradient_norm"] = energy ** 0.5
            self.input_gradient_energies.clear()
            if isinstance(self.mixer, QuantumMixer) and mixer is not None:
                for layer, gradient in enumerate(mixer.reshape_as(self.mixer.angles), 1):
                    sample[f"quantum_layer_{layer}_gradient_norm"] = gradient.norm().item()
        self.samples.append(sample)

    @torch.no_grad()
    def finish_epoch(self, seconds: float) -> dict:
        metrics = {}
        for key in self.samples[0]:
            values = [sample[key] for sample in self.samples if sample[key] is not None]
            metrics[key] = mean(values) if values else None
        self.samples.clear()
        parameters = self._parameters()
        return {
            **metrics, "epoch_seconds": seconds, "mixer_parameters": parameters.tolist(),
            "mixer_parameter_drift": (parameters - self.initial).norm().item(),
            "residual_scale": self.model.config.alpha if self.model.config.kind == "residual" else None,
        }
