"""Shared pytest fixtures and configuration."""

import torch
import pytest


@pytest.fixture(autouse=True)
def set_seed():
    """Fix random seed for reproducibility across all tests."""
    torch.manual_seed(42)
