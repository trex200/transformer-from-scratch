"""Download the exact EN→FR CSV used in the notebook (175621 pairs)."""

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "eng_-french.csv"

URL = (
    "https://raw.githubusercontent.com/"
    "SayamAlt/English-to-French-Language-Translation-using-Seq2Seq-Modeling/"
    "main/eng-french.csv"
)


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {DEST} ...")
    urlretrieve(URL, DEST)
    print(f"Done. {DEST.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
