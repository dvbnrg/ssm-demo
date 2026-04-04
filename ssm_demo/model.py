"""
Full Mamba sequence model.

Stacks N MambaBlocks with a final layer norm, producing a causal sequence
model suitable for tasks such as language modelling or time-series prediction.

Usage example
-------------
>>> from ssm_demo import MambaModel
>>> model = MambaModel(vocab_size=256, d_model=128, n_layers=4)
>>> tokens = torch.randint(0, 256, (2, 64))   # (batch, seq_len)
>>> logits = model(tokens)                     # (2, 64, 256)
"""

import torch
import torch.nn as nn

from .mamba import MambaBlock


class MambaModel(nn.Module):
    """Causal sequence model composed of stacked Mamba blocks.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (number of token embeddings).
    d_model : int
        Model dimension.
    n_layers : int
        Number of Mamba blocks to stack.
    d_state : int
        SSM state dimension for each block.
    d_conv : int
        Depthwise conv kernel size for each block.
    expand : int
        Inner expansion factor for each block.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding and output weights (common in language models)
        self.lm_head.weight = self.embedding.weight

    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor, shape (batch, seq_len)
            Token indices.

        Returns
        -------
        logits : torch.Tensor, shape (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)       # (B, L, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)              # (B, L, vocab_size)

    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
