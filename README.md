# ssm-demo
A simple repo demoing state space models (SSMs), including a vanilla SSM and a Mamba-style selective scan implementation.

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
