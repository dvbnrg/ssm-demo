"""
Mamba Block — Selective State Space Model.

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
          Gu & Dao, 2023 (https://arxiv.org/abs/2312.00752)

Architecture of a single Mamba block:

    Input  x : (B, L, d_model)
           │
           ├─── Linear ───── d_inner ──── Conv1d ──── SiLU ──── Selective-SSM ───┐
           │                                                                       ×  ── Linear ── Output
           └─── Linear ───── d_inner ──── SiLU  ───────────────────────────────────┘

The *selective* SSM (S6) differs from a plain SSM in that B, C, and Δ are
*input-dependent*: they are computed from the input at each time step, giving
the model the ability to selectively propagate or ignore information.

Selective scan:
    Ā_t = exp(Δ_t · A)
    B̄_t = Δ_t · B_t
    h_t  = Ā_t · h_{t-1} + B̄_t · x_t
    y_t  = C_t · h_t + D · x_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """The S6 core of Mamba: a selective state-space scan.

    Parameters
    ----------
    d_inner : int
        Inner channel dimension (= expand * d_model).
    d_state : int
        SSM state dimension N.
    dt_rank : int
        Rank of the Δ projection.  Mamba uses ceil(d_model / 16).
    dt_min, dt_max : float
        Log-uniform range for Δ initialisation.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_inner / 16)

        # --- A: fixed structure, learned as log|A| (kept negative / stable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(d_inner, -1)                       # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # --- D: skip-connection weight
        self.D = nn.Parameter(torch.ones(d_inner))

        # --- Projections that produce (Δ, B, C) from the input
        # x_proj maps d_inner → dt_rank + 2*d_state
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt_proj maps dt_rank → d_inner (with bias for Δ init)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # Initialise dt_proj so Δ ≈ softplus⁻¹(U[dt_min, dt_max])
        dt_init = torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        dt_init = torch.exp(dt_init)                    # (d_inner,)
        # inverse softplus: log(exp(x) - 1) ≈ x for x >> 0
        inv_softplus = torch.log(torch.expm1(dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Selective scan over the sequence.

        Parameters
        ----------
        x : torch.Tensor, shape (B, L, d_inner)

        Returns
        -------
        y : torch.Tensor, shape (B, L, d_inner)
        """
        B_batch, L, _ = x.shape

        # A is negative (stable)
        A = -torch.exp(self.A_log.float())              # (d_inner, d_state)

        # --- compute input-dependent Δ, B, C
        x_dbl = self.x_proj(x)                         # (B, L, dt_rank + 2*d_state)
        dt_raw, B_mat, C_mat = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # dt_raw: (B, L, dt_rank) → dt: (B, L, d_inner)
        dt = F.softplus(self.dt_proj(dt_raw))           # (B, L, d_inner)

        # --- discretise: zero-order-hold for A, Euler for B
        # dA: (B, L, d_inner, d_state) — broadcast dt over d_state
        dA = torch.exp(dt.unsqueeze(-1) * A)            # broadcast A: (d_inner,d_state)
        # dB: (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)      # B_mat: (B,L,d_state)

        # --- sequential selective scan
        h = x.new_zeros(B_batch, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            # dA[:,t]: (B, d_inner, d_state)
            # dB[:,t] * x[:,t,∙]: outer along d_inner × d_state
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # C_mat[:,t]: (B, d_state)  → y_t: (B, d_inner)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                      # (B, L, d_inner)
        y = y + x * self.D                              # skip connection
        return y


# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """A single Mamba residual block.

    Wraps SelectiveSSM with the gated MLP structure and a residual connection.

    Parameters
    ----------
    d_model : int
        Model (input/output) dimension.
    d_state : int
        SSM state dimension N.
    d_conv : int
        Kernel size of the depthwise causal convolution.
    expand : int
        Channel expansion factor inside the block (d_inner = expand * d_model).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model

        # Input norm + projection to 2×d_inner (two branches)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv on the SSM branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,       # left-pad for causality
            bias=True,
        )

        # Selective SSM
        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mamba block with residual connection.

        Parameters
        ----------
        x : torch.Tensor, shape (B, L, d_model)

        Returns
        -------
        out : torch.Tensor, shape (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split into two branches: SSM branch (x_ssm) and gate branch (z)
        xz = self.in_proj(x)                            # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        # --- SSM branch ---
        # 1. Causal depthwise conv
        x_ssm = x_ssm.transpose(1, 2)                  # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, : x.shape[1]] # trim padding → causal
        x_ssm = x_ssm.transpose(1, 2)                  # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # 2. Selective SSM
        x_ssm = self.ssm(x_ssm)                        # (B, L, d_inner)

        # --- Gate branch ---
        z = F.silu(z)                                   # (B, L, d_inner)

        # --- Combine and project ---
        out = self.out_proj(x_ssm * z)                  # (B, L, d_model)
        return out + residual                           # residual connection
