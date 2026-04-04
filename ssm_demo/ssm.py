"""
Vanilla State Space Model (SSM).

A linear SSM maps an input sequence u(t) to an output sequence y(t) via a
hidden state h(t) using the recurrence:

    h'(t) = A h(t) + B u(t)      (continuous-time)
    y(t)  = C h(t) + D u(t)

In the discrete-time (recurrent) form with step size Δ (ZOH discretisation):

    h_t = Ā h_{t-1} + B̄ u_t
    y_t = C h_t + D u_t

where:
    Ā = exp(Δ A)
    B̄ = (Ā - I) A⁻¹ B  ≈  Δ B   (for diagonal A this simplifies nicely)

This module implements the discretised SSM as a learnable PyTorch layer.
A and B are diagonal for efficiency (S4D parameterisation).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSM(nn.Module):
    """Learnable discrete-time diagonal SSM.

    Parameters
    ----------
    d_model : int
        Input/output feature dimension.
    d_state : int
        Dimension of the hidden state (N in the S4 paper).
    dt_min, dt_max : float
        Range for the learnable log time-step Δ (log-uniform init).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # --- A: diagonal state matrix, parameterised as log(|A|) so A stays
        #     real-negative and the system is stable.
        # Shape: (d_model, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(d_model, -1)           # (d_model, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # --- B and C: input / output projection onto the state space
        # Shape: (d_model, d_state)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # --- D: skip connection (direct term)
        self.D = nn.Parameter(torch.ones(d_model))

        # --- Δ (delta): per-feature log time-step, log-uniform in [dt_min, dt_max]
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_AB_bar(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return discretised (Ā, B̄) using the ZOH rule for diagonal A."""
        dt = torch.exp(self.log_dt)                     # (d_model,)
        A = -torch.exp(self.A_log)                      # (d_model, d_state)  negative

        # Ā = exp(Δ A)
        # dt is (d_model,), A is (d_model, d_state) → broadcast
        A_bar = torch.exp(dt.unsqueeze(-1) * A)         # (d_model, d_state)

        # B̄ = (Ā − I) / A * B  (element-wise for diagonal A)
        B_bar = (A_bar - 1.0) / A * self.B              # (d_model, d_state)

        return A_bar, B_bar

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Run the SSM over a sequence.

        Parameters
        ----------
        u : torch.Tensor, shape (batch, seq_len, d_model)

        Returns
        -------
        y : torch.Tensor, shape (batch, seq_len, d_model)
        """
        B_batch, L, d = u.shape
        assert d == self.d_model, f"Expected d_model={self.d_model}, got {d}"

        A_bar, B_bar = self._get_AB_bar()   # (d_model, d_state) each
        C = self.C                           # (d_model, d_state)
        D = self.D                           # (d_model,)

        # Initialise hidden state
        h = torch.zeros(B_batch, d, self.d_state, device=u.device, dtype=u.dtype)

        ys = []
        for t in range(L):
            u_t = u[:, t, :]                             # (batch, d_model)
            # h: (batch, d_model, d_state)
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * u_t.unsqueeze(-1)
            y_t = (h * C.unsqueeze(0)).sum(-1)           # (batch, d_model)
            y_t = y_t + D * u_t                          # skip connection
            ys.append(y_t)

        return torch.stack(ys, dim=1)                    # (batch, L, d_model)
