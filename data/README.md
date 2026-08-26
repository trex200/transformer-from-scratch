# Data

The training CSV is **not** in this repo (too big for Git).

## File used

- **Name:** `eng_-french.csv`
- **Shape:** `175621` rows × `2` columns
- **Columns:** `English words/sentences`, `French words/sentences`

Drop at `data/eng_-french.csv` and run `python notebooks/01_prepare_enfr_data.py`.

## Vocab from the notebook run

| lang | n_words |
|---|---|
| en | 14301 |
| fr | 25726 |

Decoder `Lang` (French): PAD=0, SOS=1, EOS=2, UNK=3.

Encoder `inpLang` (English): starts with only PAD=0 and UNK=1 (`n_words = 2`). Different UNK id on purpose.

Encoder ids = English tokens (missing → 1).
Decoder ids = SOS + French tokens + EOS, pad to MAX_LENGTH=64.
Batch size = 32.
