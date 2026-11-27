"""Batched, differentiable four-qubit statevector and a classical control."""

import math

import torch
from torch import Tensor, nn

QUBITS = 4
STATE_SIZE = 2**QUBITS


def encode_angles(features: Tensor) -> Tensor:
    """Map four real features to angles bounded by ±pi."""
    if features.ndim != 2 or features.shape[1] != QUBITS or features.dtype not in (torch.float32, torch.float64):
        raise ValueError("expected a (batch, 4) float32 or float64 feature matrix")
    return math.pi * features.tanh()


class QuantumMixer(nn.Module):
    """RY encoding, layers of RZ then RY, optional linear CNOT chain, Z readout."""

    def __init__(self, depth: int = 2, *, entangle: bool = True, trainable: bool = True) -> None:
        super().__init__()
        if type(depth) is not int or depth < 1:
            raise ValueError("circuit depth must be a positive integer")
        self.entangle = entangle
        self.angles = nn.Parameter(torch.empty(depth, QUBITS, 2), requires_grad=trainable)
        with torch.no_grad():
            self.angles.uniform_(-0.1, 0.1)
            self.angles[..., 1].add_(math.pi / 4)

        # Basis index b uses wire i at bit i. CNOT is its own inverse.
        basis = torch.arange(STATE_SIZE)
        wires = torch.arange(QUBITS)
        bits = (basis >> wires[:, None]) & 1
        self.register_buffer("partners", basis ^ (1 << wires[:, None]), persistent=False)
        self.register_buffer("z_signs", (1 - 2 * bits).T.to(torch.float32), persistent=False)
        self.register_buffer("cnot_indices", basis ^ (bits[:-1] << (wires[:-1, None] + 1)), persistent=False)
        self.register_buffer("initial", (basis == 0).to(torch.float32), persistent=False)

    def _ry(self, state: Tensor, wire: int, theta: Tensor) -> Tensor:
        half = theta[..., None] / 2
        other = state.index_select(1, self.partners[wire])
        return half.cos() * state - half.sin() * self.z_signs[:, wire] * other

    def _rz(self, state: Tensor, wire: int, theta: Tensor) -> Tensor:
        half = theta / 2
        phase = torch.complex(half.cos().expand(STATE_SIZE), -half.sin() * self.z_signs[:, wire])
        return state * phase

    def statevector(self, features: Tensor) -> Tensor:
        angles = encode_angles(features)
        state = self.initial.to(angles.dtype).expand(len(features), -1)
        for wire in range(QUBITS):
            state = self._ry(state, wire, angles[:, wire])
        for layer in self.angles:
            for wire in range(QUBITS):
                state = self._rz(state, wire, layer[wire, 0])
                state = self._ry(state, wire, layer[wire, 1])
            if self.entangle:
                for indices in self.cnot_indices:
                    state = state.index_select(1, indices)
        return state

    def forward(self, features: Tensor) -> Tensor:
        state = self.statevector(features)
        probabilities = state.real.square() + state.imag.square()
        expectations = probabilities @ self.z_signs.to(probabilities.dtype)
        return expectations.clamp(-1, 1)


class ClassicalMixer(nn.Module):
    """Bounded 4-by-4 mixer with 16 weights, matching the two-layer PQC count."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(QUBITS, QUBITS, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(encode_angles(features) / math.pi).tanh()
