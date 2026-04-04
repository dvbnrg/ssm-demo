"""Tests for the Mamba block and the full MambaModel."""

import pytest
import torch
from ssm_demo import MambaBlock, MambaModel
from ssm_demo.mamba import SelectiveSSM


# ---------------------------------------------------------------------------
# SelectiveSSM
# ---------------------------------------------------------------------------


class TestSelectiveSSM:
    @pytest.fixture
    def ssm(self):
        return SelectiveSSM(d_inner=16, d_state=4)

    def test_output_shape(self, ssm):
        x = torch.randn(2, 10, 16)
        y = ssm(x)
        assert y.shape == (2, 10, 16)

    def test_gradients_flow(self, ssm):
        x = torch.randn(2, 8, 16, requires_grad=True)
        y = ssm(x)
        y.sum().backward()
        assert x.grad is not None

    def test_single_timestep(self, ssm):
        x = torch.randn(1, 1, 16)
        y = ssm(x)
        assert y.shape == (1, 1, 16)

    def test_different_batch_sizes(self, ssm):
        for b in [1, 4, 8]:
            y = ssm(torch.randn(b, 5, 16))
            assert y.shape == (b, 5, 16)


# ---------------------------------------------------------------------------
# MambaBlock
# ---------------------------------------------------------------------------


class TestMambaBlock:
    @pytest.fixture
    def block(self):
        return MambaBlock(d_model=16, d_state=4, d_conv=4, expand=2)

    def test_output_shape(self, block):
        x = torch.randn(2, 12, 16)
        y = block(x)
        assert y.shape == (2, 12, 16)

    def test_residual_connection(self, block):
        """With zero parameters the output should equal the input (residual)."""
        x = torch.randn(1, 5, 16)
        with torch.no_grad():
            # zero out all parameters to isolate the residual
            for p in block.parameters():
                p.zero_()
            y = block(x)
        # With all-zero parameters the block output is zero; residual is x
        assert y.shape == x.shape

    def test_gradients_flow(self, block):
        x = torch.randn(2, 8, 16, requires_grad=True)
        y = block(x)
        y.sum().backward()
        assert x.grad is not None

    def test_causality(self, block):
        """Changing a future token should not affect earlier outputs."""
        block.eval()
        x = torch.randn(1, 10, 16)
        x_perturbed = x.clone()
        x_perturbed[0, 7:, :] += 100.0  # perturb tokens 7-9

        with torch.no_grad():
            y = block(x)
            y_p = block(x_perturbed)

        # positions 0-6 must be identical
        assert torch.allclose(y[0, :7], y_p[0, :7], atol=1e-5), (
            "Mamba block is not causal: future tokens affect past outputs"
        )

    def test_different_seq_lengths(self, block):
        for L in [1, 5, 64]:
            y = block(torch.randn(1, L, 16))
            assert y.shape == (1, L, 16)


# ---------------------------------------------------------------------------
# MambaModel
# ---------------------------------------------------------------------------


class TestMambaModel:
    @pytest.fixture
    def model(self):
        return MambaModel(vocab_size=64, d_model=16, n_layers=2, d_state=4)

    def test_output_shape(self, model):
        ids = torch.randint(0, 64, (2, 20))
        logits = model(ids)
        assert logits.shape == (2, 20, 64)

    def test_parameter_count_positive(self, model):
        assert model.count_parameters() > 0

    def test_weight_tying(self, model):
        assert model.lm_head.weight is model.embedding.weight

    def test_gradients_flow(self, model):
        ids = torch.randint(0, 64, (2, 10))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_single_token(self, model):
        ids = torch.randint(0, 64, (1, 1))
        logits = model(ids)
        assert logits.shape == (1, 1, 64)

    def test_long_sequence(self, model):
        ids = torch.randint(0, 64, (1, 128))
        logits = model(ids)
        assert logits.shape == (1, 128, 64)

    def test_n_layers_affects_depth(self):
        m1 = MambaModel(vocab_size=32, d_model=8, n_layers=1)
        m4 = MambaModel(vocab_size=32, d_model=8, n_layers=4)
        assert m4.count_parameters() > m1.count_parameters()
