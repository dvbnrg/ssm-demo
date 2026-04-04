"""
example.py — quick demonstration of the SSM and Mamba implementations.

Run with:
    python example.py
"""

import torch
import torch.nn as nn
from ssm_demo import SSM, MambaBlock, MambaModel


# ──────────────────────────────────────────────
# 1. Vanilla SSM on a random sequence
# ──────────────────────────────────────────────

print("=" * 60)
print("1. Vanilla SSM")
print("=" * 60)

ssm = SSM(d_model=32, d_state=16)
x = torch.randn(4, 64, 32)        # batch=4, seq_len=64, d_model=32
y = ssm(x)
print(f"  Input  shape : {x.shape}")
print(f"  Output shape : {y.shape}")
print(f"  Parameters   : {sum(p.numel() for p in ssm.parameters()):,}")

# ──────────────────────────────────────────────
# 2. Single Mamba block
# ──────────────────────────────────────────────

print()
print("=" * 60)
print("2. MambaBlock")
print("=" * 60)

block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)
x = torch.randn(2, 128, 64)
y = block(x)
print(f"  Input  shape : {x.shape}")
print(f"  Output shape : {y.shape}")
print(f"  Parameters   : {sum(p.numel() for p in block.parameters()):,}")

# ──────────────────────────────────────────────
# 3. Full Mamba language model
# ──────────────────────────────────────────────

print()
print("=" * 60)
print("3. MambaModel (language model)")
print("=" * 60)

model = MambaModel(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    d_state=16,
    d_conv=4,
    expand=2,
)
token_ids = torch.randint(0, 256, (2, 64))   # batch=2, seq_len=64
logits = model(token_ids)
print(f"  Token IDs shape : {token_ids.shape}")
print(f"  Logits shape    : {logits.shape}")
print(f"  Parameters      : {model.count_parameters():,}")

# ──────────────────────────────────────────────
# 4. Tiny training loop (next-token prediction)
# ──────────────────────────────────────────────

print()
print("=" * 60)
print("4. Training loop (5 steps)")
print("=" * 60)

model = MambaModel(vocab_size=64, d_model=64, n_layers=2, d_state=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

torch.manual_seed(0)
for step in range(5):
    tokens = torch.randint(0, 64, (4, 32))      # (batch, seq_len)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    logits = model(inputs)                       # (4, 31, 64)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, 64),
        targets.reshape(-1),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  step {step + 1}: loss = {loss.item():.4f}")

print()
print("Done ✓")
