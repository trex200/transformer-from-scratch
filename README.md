# Transformer from Scratch

From-scratch implementation of the Transformer architecture from **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017).

Built as a learning project: every block (embeddings, positional encoding, multi-head attention, FFN, encoder/decoder stacks, masking) is written by hand so the paper maps onto real code.

> Status: **scaffold up. Code, training outputs, and screenshots landing next.**

## Why this repo exists

I wanted to stop treating `nn.Transformer` as a black box and actually build scaled dot-product attention, multi-head attention, and the encoder-decoder stack myself.

## Architecture (paper to code)

| Paper piece | What it does |
|---|---|
| Scaled dot-product attention | `softmax(QK^T / sqrt(d_k)) V` |
| Multi-head attention | Split `d_model` across `h` heads, attend in parallel, concat + project |
| Positional encoding | Sinusoidal PE so the model knows token order |
| Encoder block | Self-attn -> add & norm -> FFN -> add & norm |
| Decoder block | Masked self-attn -> encoder-decoder attn -> FFN |
| Masks | Padding mask + causal (look-ahead) mask on the decoder |

## Repo layout

```
transformer-from-scratch/
|-- src/
|-- notebooks/
|-- assets/screenshots/
|-- data/
|-- results/
|-- requirements.txt
|-- README.md
```

## Setup

```bash
git clone https://github.com/trex200/transformer-from-scratch.git
cd transformer-from-scratch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

_To be filled._

## Results

_To be filled with screenshots and numbers._

## Paper

Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
https://arxiv.org/abs/1706.03762

## License

MIT. See LICENSE.
