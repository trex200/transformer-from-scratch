# Transformer from Scratch

From-scratch implementation of the Transformer architecture from **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017).

Learning project: build the blocks by hand, then train English → French on a sentence-pair CSV.

> Status: **data pipeline is in.** Model + training loop + output screenshots next.

## Why this repo exists

I wanted to stop treating `nn.Transformer` as a black box and actually build scaled dot-product attention, multi-head attention, and the encoder-decoder stack myself.

## What's in so far

CSV → clean text → word vocabs → SOS/EOS/PAD/UNK ids → batched tensors.

| Step | Where |
|---|---|
| Normalize + vocabs + Dataset | [`src/data.py`](src/data.py) |
| Runnable walkthrough | [`notebooks/01_prepare_enfr_data.py`](notebooks/01_prepare_enfr_data.py) |
| Dataset notes | [`data/README.md`](data/README.md) |
| Numbers from the notebook | [`results/data_prep_notes.md`](results/data_prep_notes.md) |

### Data handling

- Load `eng_-french.csv` (175621 rows).
- `unicodeToAscii` / `normalizeString` from the **PyTorch seq2seq translation tutorial**.
- Word-level vocabs from the notebook run: **en 14301**, **fr 25726**.
- Encoder ids = English tokens (UNK if missing).
- Decoder ids = `SOS + French tokens + EOS`.
- Pad / truncate to `MAX_LENGTH = 64`, batch size `32`.

Full CSV is not committed. See `data/README.md`.

## Credits

- Vaswani et al., 2017. *Attention Is All You Need*.
- Text cleanup: [PyTorch seq2seq tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- Sentence pairs: Tatoeba / manythings.org Anki, commonly packaged as Kaggle `eng_-french.csv`

## License

MIT. See [LICENSE](LICENSE).
