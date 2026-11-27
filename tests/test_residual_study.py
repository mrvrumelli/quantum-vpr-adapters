import unittest
from dataclasses import replace

import numpy as np
import torch

from vpr_baseline.adapters import AdapterConfig, MLPAdapter
from vpr_baseline.cache import DescriptorSet, SplitCache
from vpr_baseline.data import ImageRecord
from vpr_baseline.mixer_monitor import MixerMonitor
from vpr_baseline.residual_study import feature_covariance, measure_snapshot, paired_block_bootstrap
from vpr_baseline.retrieval import l2_normalize


class ResidualStudyTests(unittest.TestCase):
    def test_identity_mixer_preserves_matched_projections_and_gradients(self):
        config = AdapterConfig("residual", 64, up_bias=False, activation="leaky_relu", transform="quantum")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            quantum = MLPAdapter(config)
            torch.manual_seed(42)
            identity = MLPAdapter(replace(config, transform="identity"))
        self.assertEqual(len(tuple(identity.mixer.parameters())), 0)
        for name, value in identity.state_dict().items():
            torch.testing.assert_close(value, quantum.state_dict()[name], atol=0, rtol=0)
        monitor = MixerMonitor(identity, detailed=True)
        inputs = torch.eye(512)[:4]
        projected, output, _, _ = identity.components(inputs)
        torch.testing.assert_close(projected, output, atol=0, rtol=0)
        identity(inputs)[:, :10].sum().backward()
        monitor.observe()
        report = monitor.finish_epoch(0.0)
        self.assertGreater(report["mixer_input_gradient_norm"], 0)
        self.assertIsNone(report["mixer_gradient_norm"])
        self.assertEqual(report["mixer_parameters"], [])

    def test_quantum_monitor_measures_bottleneck_input_and_layer_gradients(self):
        for trainable in (True, False):
            rng = torch.Generator().manual_seed(9)
            model = MLPAdapter(AdapterConfig("residual", transform="quantum", train_quantum=trainable))
            monitor = MixerMonitor(model, detailed=True)
            inputs = l2_normalize(torch.randn(4, 512, generator=rng))
            bottleneck, _, _, raw = model.components(inputs)
            bottleneck.retain_grad()
            raw[:, :10].sum().backward()
            monitor.observe()
            report = monitor.finish_epoch(0.0)
            self.assertAlmostEqual(report["mixer_input_gradient_norm"], bottleneck.grad.norm().item())
            self.assertGreater(report["mixer_input_gradient_norm"], 0)
            if trainable:
                for i, gradient in enumerate(model.mixer.angles.grad, 1):
                    self.assertAlmostEqual(report[f"quantum_layer_{i}_gradient_norm"], gradient.norm().item())
            else:
                self.assertNotIn("quantum_layer_1_gradient_norm", report)
                self.assertIsNone(model.mixer.angles.grad)

    def test_covariance_and_zero_scale_snapshot(self):
        features = torch.tensor([[-1.0, 4.0], [1.0, 4.0]])
        covariance, rank = feature_covariance(features)
        torch.testing.assert_close(covariance, torch.diag(torch.tensor([1., 0.], dtype=torch.float64)))
        self.assertEqual(rank, 1)
        self.assertEqual(feature_covariance(features[:1].expand(3, -1))[1], 0)
        inputs = torch.eye(512)[:2]
        images = tuple(ImageRecord(str(i), f"Image{i}.jpg") for i in range(2))
        cache = SplitCache("val", {}, DescriptorSet(images, inputs), DescriptorSet(images, inputs.flip(0)))
        positives = [{1}, {0}]
        baseline = measure_snapshot(cache, positives, None)
        adapter = MLPAdapter(AdapterConfig("residual", alpha=0, transform="quantum"))
        zero = measure_snapshot(cache, positives, adapter)
        torch.testing.assert_close(zero.ranks, baseline.ranks, atol=0, rtol=0)
        self.assertEqual(zero.metrics["MRR"], 1)
        self.assertEqual(zero.metrics["scaled_correction_mean"], 0)
        self.assertGreater(zero.metrics["unscaled_correction_mean"], 0)
        self.assertEqual(zero.metrics["cosine_to_original_mean"], 1)

    def test_block_bootstrap_keeps_contiguous_pairs_and_frame_order(self):
        values = np.array([[1., -1., 2., -2.], [3., -3., 4., -4.]])
        frames = (90, 91, 92, 93)
        self.assertEqual(paired_block_bootstrap(values, frames, block_length=2), (0, 0))
        self.assertEqual(paired_block_bootstrap(values[:, [2, 1, 0, 3]], (92, 91, 90, 93), block_length=2), (0, 0))
        self.assertEqual(paired_block_bootstrap(values + 1, frames, block_length=2), (1, 1))
        with self.assertRaisesRegex(ValueError, "consecutive"):
            paired_block_bootstrap(values, (90, 91, 93, 94), block_length=2)
