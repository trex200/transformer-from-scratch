# Data

The training CSV is **not** in this repo (too big for Git).

## File used

- **Name:** `eng_-french.csv`
- **Shape seen in the notebook:** `175621` rows × `2` columns
- **Columns:** `English words/sentences`, `French words/sentences`
- **First rows:** `Hi.` / `Salut!`, `Run!` / `Cours !`

Common Kaggle dump of Tatoeba / manythings.org Anki pairs:
https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench

Drop the CSV at `data/eng_-french.csv` and run:

```bash
python notebooks/01_prepare_enfr_data.py
```

## Vocab from the notebook run

| lang | n_words |
|---|---|
| en | 14301 |
| fr | 25726 |

| token | id |
|---|---|
| PAD | 0 |
| UNK | 1 |
| SOS | 2 |
| EOS | 3 |

Encoder = English tokens (UNK if missing).
Decoder = SOS + French tokens + EOS, pad to MAX_LENGTH=64.
Batch size = 32.
