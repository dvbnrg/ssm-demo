"""Tests for the vanilla SSM."""

import pytest
import torch
from ssm_demo import SSM


@pytest.fixture
def ssm():
    return SSM(d_model=8, d_state=4)


def test_output_shape(ssm):
    u = torch.randn(2, 10, 8)
    y = ssm(u)
    assert y.shape == (2, 10, 8), f"Expected (2,10,8), got {y.shape}"


def test_batch_size_one(ssm):
    u = torch.randn(1, 5, 8)
    y = ssm(u)
    assert y.shape == (1, 5, 8)


def test_single_timestep(ssm):
    u = torch.randn(3, 1, 8)
    y = ssm(u)
    assert y.shape == (3, 1, 8)


def test_gradients_flow(ssm):
    u = torch.randn(2, 6, 8, requires_grad=True)
    y = ssm(u)
    loss = y.sum()
    loss.backward()
    assert u.grad is not None
    assert not torch.all(u.grad == 0)


def test_parameters_are_learned():
    ssm = SSM(d_model=4, d_state=2)
    param_names = {n for n, _ in ssm.named_parameters()}
    assert "A_log" in param_names
    assert "B" in param_names
    assert "C" in param_names
    assert "D" in param_names
    assert "log_dt" in param_names


def test_a_bar_stable():
    """Ā = exp(Δ·A) should have |entries| ≤ 1 for a stable system."""
    ssm = SSM(d_model=4, d_state=4)
    A_bar, _ = ssm._get_AB_bar()
    assert (A_bar.abs() <= 1.0 + 1e-6).all(), "SSM is not stable (|Ā| > 1)"


def test_wrong_d_model_raises(ssm):
    u = torch.randn(2, 5, 16)  # wrong last dim
    with pytest.raises(AssertionError):
        ssm(u)


def test_different_seq_lengths(ssm):
    for seq_len in [1, 7, 32, 100]:
        y = ssm(torch.randn(1, seq_len, 8))
        assert y.shape == (1, seq_len, 8)
