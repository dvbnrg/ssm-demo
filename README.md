# ssm-demo
A simple repo demoing state space models (SSMs), including a vanilla SSM and a Mamba-style selective scan implementation.

## Notebook use case — causal language modelling

`ssm_demo.ipynb` demonstrates the full SSM → Mamba progression applied to **next-token prediction** (causal language modelling).  Starting from random token sequences, it builds up four components step by step:

1. **Vanilla SSM** — a linear recurrence with fixed parameters that maps an input sequence to an output sequence of the same shape.
2. **Selective SSM (S6)** — the core Mamba innovation: Δ, B, and C become functions of the input at each position, so the model can selectively remember or ignore each token.
3. **Mamba Block** — wraps the Selective SSM with LayerNorm, a causal depthwise Conv1d (local context), SiLU gating, and a residual connection.
4. **Mamba Language Model** — stacks N Mamba Blocks between a token embedding table and a linear output head (with weight tying) to produce next-token logits.

The notebook closes with a **mini training loop** that trains the language model on random token sequences using the standard teacher-forcing / cross-entropy objective, verifying that the full computation graph — from the embedding through all recurrent layers to the LM head — is differentiable and trainable end-to-end.

## Use Cases

State space models — and the Mamba selective-scan variant in particular — are well-suited for any task that involves **long, ordered sequences** where efficient, linear-time processing matters.

| Domain | Application |
|---|---|
| **Language modelling** | Next-token prediction and text generation, as shown in `example.py`. Mamba scales to much longer contexts than Transformer-based models at the same compute budget. |
| **Time-series forecasting** | Predicting future values in financial data, weather signals, or IoT sensor streams, where the recurrent structure naturally captures long-range temporal dependencies. |
| **Audio & speech** | Processing raw waveforms or mel-spectrograms for speech recognition, music generation, and audio classification. SSMs handle the very long sequences produced by high sample-rate audio. |
| **Genomics & biology** | Modelling DNA/RNA sequences and protein sequences, which can be tens of thousands of tokens long — a regime where attention-based methods become prohibitively expensive. |
| **Video understanding** | Treating video as a sequence of frames (or patches) to perform action recognition, anomaly detection, or video captioning with sub-quadratic complexity. |
| **Control & robotics** | Modelling dynamical systems for model-predictive control or reinforcement learning, where the SSM's explicit state-transition structure mirrors physical state equations. |

## Prerequisites

- Python 3.9+
- [pip](https://pip.pypa.io/en/stable/)

## Installation

```bash
pip install -r requirements.txt
```

## Running the example

The `example.py` script walks through four demos: a vanilla SSM, a single Mamba block, a full Mamba language model, and a tiny training loop.

```bash
python example.py
```

## Running the tests

```bash
pip install pytest          # if not already installed
pytest
```
