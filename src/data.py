"""
English → French data pipeline for the from-scratch Transformer.

Turns the pandas CSV (`eng_-french.csv`) into padded encoder / decoder
tensors with SOS / EOS / PAD / UNK.

`unicodeToAscii` and `normalizeString` follow the PyTorch seq2seq
translation tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
INP_UNK_token = 1  # encoder-side UNK lives at id 1 (see inpLang)

MAX_LENGTH = 64
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EN_COL = "English words/sentences"
FR_COL = "French words/sentences"


class Lang:
    """French / decoder vocab. Reserved: PAD=0, SOS=1, EOS=2, UNK=3."""

    def __init__(self, name):
        self.name = name
        self.word2index = {
            "<pad>": PAD_token,
            "<sos>": SOS_token,
            "<eos>": EOS_token,
            "<unk>": UNK_token,
        }
        self.word2count = {
            "<pad>": 0,
            "<sos>": 0,
            "<eos>": 0,
            "<unk>": 0,
        }
        self.index2word = {
            PAD_token: "<pad>",
            SOS_token: "<sos>",
            EOS_token: "<eos>",
            UNK_token: "<unk>",
        }
        self.n_words = 4  # PAD, SOS, EOS, UNK

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            if word.strip():
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class inpLang:
    """English / encoder vocab. Only PAD=0 and UNK=1 at start (different UNK id)."""

    def __init__(self, name):
        self.name = name
        self.word2index = {
            "<pad>": PAD_token,
            "<unk>": INP_UNK_token,
        }
        self.word2count = {
            "<pad>": 0,
            "<unk>": 0,
        }
        self.index2word = {
            PAD_token: "<pad>",
            INP_UNK_token: "<unk>",
        }
        self.n_words = 2  # only PAD and UNK initially

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            if word.strip():
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalizeString(s: str) -> str:
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z' .!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s)
    return s.strip()


def indexesFromSentence(lang, sentence):
    return (
        [SOS_token]
        + [lang.word2index.get(word, UNK_token) for word in sentence.split(" ") if word.strip()]
        + [EOS_token]
    )


def indexesFromSentence_forenc(lang, sentence):
    return [
        lang.word2index.get(word, INP_UNK_token)
        for word in sentence.split(" ")
        if word.strip()
    ]


def tensorFromSentence(lang, sentence, device=DEVICE):
    indexes = indexesFromSentence(lang, sentence)[:MAX_LENGTH]
    indexes += [PAD_token] * (MAX_LENGTH - len(indexes))
    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensorFromSentence_forenc(lang, sentence, device=DEVICE):
    indexes = indexesFromSentence_forenc(lang, sentence)[:MAX_LENGTH]
    indexes += [PAD_token] * (MAX_LENGTH - len(indexes))
    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensorsFromPair(pair, input_lang, output_lang, device=DEVICE):
    return (
        tensorFromSentence_forenc(input_lang, pair[0], device=device),
        tensorFromSentence(output_lang, pair[1], device=device),
    )


class FrenchEnData(Dataset):
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return tensorsFromPair(self.pairs[idx], self.input_lang, self.output_lang)


def build_pairs_and_vocabs(df, en_col=EN_COL, fr_col=FR_COL):
    input_lang = inpLang("en")
    output_lang = Lang("fr")
    pairs = []
    input_sen = []
    output_sen = []

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


def make_dataloader(pairs, input_lang, output_lang, batch_size=BATCH_SIZE):
    dataset = FrenchEnData(pairs, input_lang, output_lang)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
