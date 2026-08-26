# Sample translations (greedy decode)

From the notebook after 10 epochs. Function: `src/translate.py`.

```
English Input:
sometimes i feel like killing me and not live anymore

French Translation:
parfois j'ai l'impression de me tuer et de ne plus vivre .
```

```python
french_output = translate_sentence(
    model=model,
    sentence=english_sentence,
    input_lang=input_lang,
    output_lang=output_lang,
    max_len=64,
    device=device,
)
```
