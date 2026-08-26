"""Notebook-faithful walkthrough: CSV → normalized pairs → vocabs → DataLoader."""

from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DEVICE, build_pairs_and_vocabs, make_dataloader

CSV_CANDIDATES = [
    ROOT / "data" / "eng_-french.csv",
    ROOT / "eng_-french.csv",
    Path("eng_-french.csv"),
]


def find_csv() -> Path:
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Put eng_-french.csv in data/ (not committed — see data/README.md)."
    )


def main() -> None:
    print("device:", DEVICE)
    csv_path = find_csv()
    df = pd.read_csv(csv_path)
    print(df.shape)
    print(df.head())

    pairs, input_lang, output_lang, _, _ = build_pairs_and_vocabs(df)

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print("\nFirst 5 normalized pairs:")
    for i in range(min(5, len(pairs))):
        print(pairs[i])

    loader = make_dataloader(pairs, input_lang, output_lang)
    print("\nSample batch shapes:")
    for batch in loader:
        print(batch[0].shape, batch[1].shape)
        print("Input example:", batch[0][0][:20])
        print("Target example:", batch[1][0][:20])
        break


if __name__ == "__main__":
    main()
