import unittest

import torch

from vpr_baseline.adapters import AdapterConfig, MLPAdapter
from vpr_baseline.retrieval import ZeroDescriptorError, l2_normalize


class AdapterTests(unittest.TestCase):
    def setUp(self):
        rng = torch.Generator().manual_seed(22)
        self.z = l2_normalize(torch.randn(3, 512, generator=rng))

    def test_forward_formulas_shapes_and_normalization(self):
        direct = MLPAdapter(AdapterConfig("direct"))
        residual = MLPAdapter(AdapterConfig("residual", alpha=0.25))
        residual.load_state_dict(direct.state_dict())
        self.assertEqual(
            [tuple(layer.weight.shape) for layer in (direct.down, direct.mlp[0], direct.up)],
            [(8, 512), (4, 8), (512, 4)],
        )
        delta = direct.up(direct.mlp(direct.down(self.z)))
        for model, expected in ((direct, delta), (residual, self.z + 0.25 * delta)):
            with self.subTest(kind=model.config.kind):
                vectors = model(self.z)
                self.assertEqual(vectors.shape, (3, 512))
                torch.testing.assert_close(vectors, l2_normalize(expected))
                torch.testing.assert_close(vectors.norm(dim=1), torch.ones(3))

    def test_residual_alpha_zero_is_normalized_identity(self):
        residual = MLPAdapter(AdapterConfig("residual", down_dim=6, hidden_dim=2, alpha=0))
        self.assertEqual(residual.mlp[0].weight.shape, (2, 6))
        torch.testing.assert_close(residual(self.z), self.z)
        with self.assertRaisesRegex(ValueError, r"\(batch, 512\)"):
            residual(self.z[:, :-1])

    def test_direct_without_up_bias_keeps_the_same_seeded_weights(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            control = MLPAdapter(AdapterConfig("direct"))
            torch.random.default_generator.manual_seed(42)
            model = MLPAdapter(AdapterConfig("direct", up_bias=False))
        self.assertIsNone(model.up.bias)
        for name, weights in model.state_dict().items():
            torch.testing.assert_close(weights, control.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            model.mlp[0].bias.fill_(1)
        torch.testing.assert_close(model(self.z).norm(dim=1), torch.ones(3))

    def test_bias_free_direct_rejects_zero_norm_output(self):
        model = MLPAdapter(AdapterConfig("direct", up_bias=False))
        with torch.no_grad():
            model.mlp[0].weight.zero_()
            model.mlp[0].bias.fill_(-1)
        with self.assertRaises(ZeroDescriptorError):
            model(self.z)

    def test_leaky_relu_keeps_negative_activations_and_shared_weights(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            relu = MLPAdapter(AdapterConfig("direct", up_bias=False))
            torch.random.default_generator.manual_seed(42)
            leaky = MLPAdapter(AdapterConfig("direct", up_bias=False, activation="leaky_relu"))
        for name, weights in relu.state_dict().items():
            torch.testing.assert_close(leaky.state_dict()[name], weights, rtol=0, atol=0)
        with torch.no_grad():
            leaky.mlp[0].weight.zero_()
            leaky.mlp[0].bias.fill_(-1)
        hidden = leaky.mlp(leaky.down(self.z))
        torch.testing.assert_close(hidden, torch.full((3, 4), -0.01))
        hidden.sum().backward()
        torch.testing.assert_close(leaky.mlp[0].bias.grad, torch.full((4,), 0.03))
        torch.testing.assert_close(leaky(self.z).norm(dim=1), torch.ones(3))

    def test_leaky_relu_still_reports_an_exact_zero_output(self):
        model = MLPAdapter(AdapterConfig("direct", up_bias=False, activation="leaky_relu"))
        with torch.no_grad():
            model.mlp[0].weight.zero_()
            model.mlp[0].bias.fill_(-1)
            model.up.weight.zero_()
        with self.assertRaises(ZeroDescriptorError) as failure:
            model(self.z)
        self.assertEqual((failure.exception.zero_rows, failure.exception.batch_rows), (3, 3))

    def test_residual_keeps_the_relu_architecture(self):
        with self.assertRaisesRegex(ValueError, "only for the Direct adapter"):
            AdapterConfig("residual", activation="leaky_relu")

    def test_gradients_reach_each_trainable_layer(self):
        for kind in ("direct", "residual"):
            with self.subTest(kind=kind), torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(5)
                model = MLPAdapter(AdapterConfig(kind))
                with torch.no_grad():
                    model.mlp[0].bias.fill_(1.0)
                model(self.z)[:, 0].mean().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
