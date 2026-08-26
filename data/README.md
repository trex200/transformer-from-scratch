# Data

Exact file used in the notebook: **`eng_-french.csv`**

| | |
|---|---|
| Rows | 175621 + header |
| Columns | `English words/sentences`, `French words/sentences` |
| First pairs | `Hi.` / `Salut!`, `Run!` / `Cours !`, `Run!` / `Courez !` |
| Size | ~12 MB |

## Source (Kaggle)

**[Language Translation (English-French)](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)**  
Author: Devicharith · License: CC0

Tatoeba / [manythings.org Anki](http://www.manythings.org/anki/) EN–FR dump with those two column names.

A 200-row peek is in [`eng_-french.sample.csv`](eng_-french.sample.csv).

Get the full file (same 175621 rows):

```bash
python scripts/fetch_enfr_data.py
```

Writes `data/eng_-french.csv`.

## Vocab from the notebook run

| lang | n_words |
|---|---|
| en | 14301 |
| fr | 25726 |

Decoder `Lang`: PAD=0, SOS=1, EOS=2, UNK=3  
Encoder `inpLang`: PAD=0, UNK=1 at start  
`MAX_LENGTH=64`, batch=32.
