# Transformer from Scratch

From-scratch implementation of the Transformer from **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017).

English → French on a sentence-pair CSV. Blocks written by hand — no `nn.Transformer`.

> Status: **v1 landed.** Data pipeline, model, 10-epoch train (avg loss 2.83 → 0.49), greedy decode.

## What's in here

| Step | Where |
|---|---|
| Normalize + vocabs + Dataset | [`src/data.py`](src/data.py) |
| Attention, PE, encoder, decoder, TranslateModel | [`src/model.py`](src/model.py) |
| Train loop | [`src/train.py`](src/train.py) |
| Greedy decode | [`src/translate.py`](src/translate.py) |
| Data notes | [`data/README.md`](data/README.md) |
| 10-epoch loss | [`results/training_notes.md`](results/training_notes.md) |
| Sample output | [`results/sample_translations.md`](results/sample_translations.md) |

## Data

`eng_-french.csv`, 175621 pairs. Vocab after normalize: **en 14301**, **fr 25726**. CSV is not committed (too big). See `data/README.md`.

`unicodeToAscii` / `normalizeString` follow the [PyTorch seq2seq tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

Encoder `inpLang`: PAD=0, UNK=1. Decoder `Lang`: PAD=0, SOS=1, EOS=2, UNK=3. `MAX_LENGTH=64`, batch=32.

## Model

| | |
|---|---|
| d_model | 256 |
| layers | 4 enc + 4 dec |
| heads | 8 |
| d_ff | 1024 |
| dropout | 0.1 |
| PE | sinusoidal buffer, max_len 64 |

Post-norm residual blocks. GELU FFN. Scaled dot-product attention with pad / causal masks.

## Train

AdamW `3e-4`, `CrossEntropyLoss(ignore_index=PAD)`, teacher forcing (`tgt[:, :-1]` → `tgt[:, 1:]`).

| epoch | avg loss |
|---|---|
| 1 | 2.83 |
| 5 | 0.77 |
| 10 | 0.49 |

~1m10s / epoch at ~77 it/s on CUDA. Checkpoints stayed local (`model_epoch_*.pt`).

## Sample (greedy)

```
EN: sometimes i feel like killing me and not live anymore
FR: parfois j'ai l'impression de me tuer et de ne plus vivre .
```

## License

MIT
