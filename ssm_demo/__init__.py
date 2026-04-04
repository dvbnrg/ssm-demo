"""
ssm_demo: A demonstration of State Space Models (SSMs) with a Mamba-style
selective scan implementation.

Modules:
    ssm   - Vanilla continuous-time / discrete-time SSM
    mamba - Mamba block (selective state-space model)
    model - Full sequence model built from Mamba blocks
"""

from .ssm import SSM
from .mamba import MambaBlock
from .model import MambaModel

__all__ = ["SSM", "MambaBlock", "MambaModel"]
