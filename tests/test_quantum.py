import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.func import functional_call

from vpr_baseline.adapters import AdapterConfig, MLPAdapter, evaluate_adapter, load_adapter
from vpr_baseline.cache import DescriptorSet, SplitCache
from vpr_baseline.data import ImageRecord
from vpr_baseline.quantum import QuantumMixer, encode_angles
from vpr_baseline.retrieval import l2_normalize
from vpr_baseline.training import TrainConfig, train_adapter


class QuantumTests(unittest.TestCase):
    def test_encoding_entanglement_and_measurement_on_known_states(self):
        features = torch.zeros((2, 4), dtype=torch.float64)
        features[1, 0] = math.atanh(0.5)
        torch.testing.assert_close(encode_angles(features)[1, 0], features.new_tensor(math.pi / 2))
        coupled = QuantumMixer(depth=1).double()
        separate = QuantumMixer(depth=1, entangle=False).double()
        with torch.no_grad():
            coupled.angles.zero_()
            separate.angles.zero_()
        torch.testing.assert_close(coupled(features), features.new_tensor([[1, 1, 1, 1], [0, 0, 0, 0]]))
        torch.testing.assert_close(separate(features), features.new_tensor([[1, 1, 1, 1], [0, 1, 1, 1]]))
        with torch.no_grad():
            separate.angles[0, 0, 0] = 0.6
        state = separate.statevector(features)[1]
        expected = torch.tensor([complex(math.cos(0.3), -math.sin(0.3)),
                                 complex(math.cos(0.3), math.sin(0.3))], dtype=torch.complex128) / math.sqrt(2)
        torch.testing.assert_close(state[:2], expected)
        torch.testing.assert_close(state.abs().square().sum(), features.new_tensor(1))

    def test_statevector_gradient_matches_finite_differences(self):
        circuit = QuantumMixer(depth=2).double()
        features = torch.tensor([[0.13, -0.27, 0.31, -0.39]], dtype=torch.float64, requires_grad=True)
        angles = torch.linspace(-0.4, 0.6, 16, dtype=torch.float64).reshape(2, 4, 2).requires_grad_()

        def evaluate(values, parameters):
            return functional_call(circuit, {"angles": parameters}, (values,))

        self.assertTrue(torch.autograd.gradcheck(evaluate, (features, angles), eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_seeded_batched_adapters_have_normalized_shapes_and_gradients(self):
        rng = torch.Generator().manual_seed(19)
        inputs = l2_normalize(torch.randn(3, 512, generator=rng))
        with torch.random.fork_rng(devices=[]):
            for kind in ("direct", "residual"):
                config = AdapterConfig(kind, 64, up_bias=False, activation="leaky_relu", transform="quantum")
                torch.random.default_generator.manual_seed(42)
                first = MLPAdapter(config)
                torch.random.default_generator.manual_seed(42)
                second = MLPAdapter(config)
                result = first(inputs)
                torch.testing.assert_close(result, second(inputs), rtol=0, atol=0)
                torch.testing.assert_close(result, torch.cat([first(row) for row in inputs.split(1)]),
                                           atol=1e-7, rtol=1e-5)
                projected, measured, delta, raw = first.components(inputs)
                self.assertEqual(projected.shape, measured.shape)
                self.assertEqual(measured.shape, (3, 4))
                self.assertEqual(delta.shape, raw.shape)
                self.assertEqual(raw.shape, (3, 512))
                self.assertTrue((measured.abs() <= 1).all())
                torch.testing.assert_close(result.norm(dim=1), torch.ones(3))
                if kind == "residual":
                    self.assertGreater((result * inputs).sum(1).min().item(), 0.99)
                (result * torch.randn(result.shape, generator=rng)).mean().backward()
                for parameter in (first.mixer.angles, first.down.weight, first.mlp[0].weight):
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.norm().item(), 0)

    def test_frozen_quantum_weights_keep_input_gradients(self):
        circuit = QuantumMixer(depth=1, trainable=False)
        features = torch.tensor([[0.13, -0.27, 0.31, -0.39]], requires_grad=True)
        circuit(features).sum().backward()
        self.assertIsNone(circuit.angles.grad)
        self.assertGreater(features.grad.norm().item(), 0)

    def test_train_save_and_evaluate_quantum_checkpoint(self):
        rng = torch.Generator().manual_seed(18)
        caches = {}
        for split, offset in (("train", 0), ("val", 40)):
            groups = {}
            for role in ("query", "gallery"):
                records = tuple(ImageRecord(f"{split}-{role}-{frame}", f"{split}/{role}/Image{frame:03d}.jpg")
                                for frame in range(offset, offset + 30, 10))
                groups[role] = DescriptorSet(records, l2_normalize(torch.randn(3, 512, generator=rng)))
            caches[split] = SplitCache(split, {"descriptor_dim": 512}, **groups)
        positives = [{0}, {1}, {2}]
        config = AdapterConfig("residual", 64, up_bias=False, activation="leaky_relu", transform="quantum")
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            history = train_adapter(config, TrainConfig(epochs=2, patience=2, seed=42), caches["train"], positives,
                                    caches["val"], positives, output)
            restored, epoch = load_adapter(output / "best.pt", caches["val"])
            self.assertEqual(restored.config, config)
            metrics = evaluate_adapter(restored, caches["val"], positives, include_mrr=True)
            self.assertGreater(metrics.pop("MRR"), 0)
            self.assertEqual(metrics, history["best_validation_recall"])
            diagnostic = history["history"][epoch - 1]["mixer_diagnostics"]
            self.assertEqual(restored.mixer.angles.flatten().tolist(), diagnostic["mixer_parameters"])
            self.assertEqual(diagnostic["residual_scale"], 0.1)
            self.assertGreater(diagnostic["mixer_gradient_norm"], 0)
