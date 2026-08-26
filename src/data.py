"""
English → French data pipeline for the from-scratch Transformer.

Turns the pandas CSV (`eng_-french.csv`) into padded encoder / decoder
tensors with SOS / EOS / PAD / UNK.

`unicodeToAscii` and `normalizeString` follow the PyTorch seq2seq
translation tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
(the unicode→ASCII helper is the well-known NFD strip from
https://stackoverflow.com/a/518232/2809427)
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

MAX_LENGTH = 64
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EN_COL = "English words/sentences"
FR_COL = "French words/sentences"


class Lang:
    """Word-level vocab. Reserved ids: PAD=0, UNK=1, SOS=2, EOS=3."""

    def __init__(self, name: str):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            UNK_token: "UNK",
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.n_words = 4

    def addSentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            if word:
                self.addWord(word)

    def addWord(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Notebook used `inpLang` for English and `Lang` for French.
inpLang = Lang


def unicodeToAscii(s: str) -> str:
    """Turn a Unicode string to plain ASCII.

    From the PyTorch seq2seq tutorial, via
    https://stackoverflow.com/a/518232/2809427
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalizeString(s: str) -> str:
    """Lowercase, trim, keep letters / apostrophes / .!? for tokenization.

    From the PyTorch seq2seq translation tutorial (adapted).
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z' .!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s)
    return s.strip()


def indexesFromSentence(lang: Lang, sentence: str) -> List[int]:
    """Decoder-side: SOS + tokens + EOS. Unknown words → UNK."""
    return (
        [SOS_token]
        + [lang.word2index.get(word, UNK_token) for word in sentence.split(" ") if word.strip()]
        + [EOS_token]
    )


def indexesFromSentence_forenc(lang: Lang, sentence: str) -> List[int]:
    """Encoder-side English: tokens only, safe UNK handling."""
    return [
        lang.word2index.get(word, UNK_token)
        for word in sentence.split(" ")
        if word.strip()
    ]


def tensorFromSentence(lang: Lang, sentence: str, device=DEVICE) -> torch.Tensor:
    indexes = indexesFromSentence(lang, sentence)[:MAX_LENGTH]
    indexes += [PAD_token] * (MAX_LENGTH - len(indexes))
    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensorFromSentence_forenc(lang: Lang, sentence: str, device=DEVICE) -> torch.Tensor:
    indexes = indexesFromSentence_forenc(lang, sentence)[:MAX_LENGTH]
    indexes += [PAD_token] * (MAX_LENGTH - len(indexes))
    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensorsFromPair(
    pair: List[str],
    input_lang: Lang,
    output_lang: Lang,
    device=DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        tensorFromSentence_forenc(input_lang, pair[0], device=device),
        tensorFromSentence(output_lang, pair[1], device=device),
    )


class FrenchEnData(Dataset):
    """Pairs → (encoder_ids [MAX_LENGTH], decoder_ids [MAX_LENGTH])."""

    def __init__(self, pairs: List[List[str]], input_lang: Lang, output_lang: Lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return tensorsFromPair(self.pairs[idx], self.input_lang, self.output_lang)


def build_pairs_and_vocabs(df, en_col: str = EN_COL, fr_col: str = FR_COL):
    """Walk the CSV rows, normalize, build vocabs. Same loop as the notebook."""
    input_lang = inpLang("en")
    output_lang = Lang("fr")
    pairs: List[List[str]] = []
    input_sen: List[str] = []
    output_sen: List[str] = []

    for _, row in df.iterrows():
        en_sentence = normalizeString(str(row[en_col]))
        fr_sentence = normalizeString(str(row[fr_col]))
        if en_sentence and fr_sentence:
            input_sen.append(en_sentence)
            output_sen.append(fr_sentence)
            pairs.append([en_sentence, fr_sentence])
            output_lang.addSentence(fr_sentence)
            input_lang.addSentence(en_sentence)

    return pairs, input_lang, output_lang, input_sen, output_sen


def make_dataloader(pairs, input_lang, output_lang, batch_size: int = BATCH_SIZE):
    dataset = FrenchEnData(pairs, input_lang, output_lang)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
