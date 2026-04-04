# ssm-demo
A simple repo demoing state space models (SSMs), including a vanilla SSM and a Mamba-style selective scan implementation.

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
