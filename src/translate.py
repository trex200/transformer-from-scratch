"""Greedy decode: English string → French string."""

import math

import torch


def translate_sentence(
    model,
    sentence,
    input_lang,
    output_lang,
    max_len=64,
    device="cuda",
):
    """Translate English sentence to French. Uses greedy decoding (argmax)."""
    model.eval()

    tokens = []
    for word in sentence.split():
        if word in input_lang.word2index:
            tokens.append(input_lang.word2index[word])
        else:
            tokens.append(3)

    src_tensor = torch.tensor([tokens], device=device)

    with torch.no_grad():
        src_mask = model.create_src_mask(src_tensor)
        src_emb = model.src_embedding(src_tensor) * math.sqrt(model.d_model)
        src_emb = src_emb + model.position_embedd(src_tensor.size(1))
        enc_output = model.encoder(src_emb, src_mask)

        tgt_tokens = [1]  # SOS = 1
        prev_token = None

        for step in range(max_len):
            tgt_tensor = torch.tensor([tgt_tokens], device=device)
            tgt_mask = model.create_tgt_mask(tgt_tensor)
            tgt_emb = model.tgt_embedding(tgt_tensor) * math.sqrt(model.d_model)
            tgt_emb = tgt_emb + model.position_embedd(tgt_tensor.size(1))
            dec_output = model.decoder(enc_output, tgt_emb, tgt_mask, src_mask)
            logits = model.final_linear(dec_output)
            next_token = logits[0, -1, :].argmax(dim=-1).item()

            if next_token == 2:
                break
            if next_token == prev_token:
                break
            if step >= src_tensor.size(1) + 10:
                break

            tgt_tokens.append(next_token)
            prev_token = next_token

        translated_words = []
        for token in tgt_tokens[1:]:
            if token == 2:
                break
            word = output_lang.index2word.get(token, "<unk>")
            translated_words.append(word)

    return " ".join(translated_words)
