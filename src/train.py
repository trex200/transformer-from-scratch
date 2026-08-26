"""Training loop from the notebook.

Teacher forcing: decoder sees tgt[:, :-1], loss is against tgt[:, 1:].
PAD positions are ignored in CrossEntropyLoss.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import PAD_token
from src.model import TranslateModel


def train(model, dataloader, epochs=10, lr=0.0003, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    print("Training Started...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for src, tgt in progress:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished | Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pt")

    print("\nTraining Completed!")
    return model


def build_model(src_vocab_size, tgt_vocab_size, max_seq_len=64, device="cuda"):
    model = TranslateModel(
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=8,
        dff=1024,
        max_seq_len=max_seq_len,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    return model.to(device)
