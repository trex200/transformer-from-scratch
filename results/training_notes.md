# Training run (notebook)

10 epochs, AdamW `lr=3e-4`, batch 32, `CrossEntropyLoss(ignore_index=PAD)`.

5489 steps / epoch.

## Hyperparams

- d_model 256, 4 layers, 8 heads, d_ff 1024, dropout 0.1
- src vocab 14301, tgt vocab 25726, max_seq_len 64, pad 0

## Avg loss

| epoch | avg loss | last-batch loss |
|---|---|---|
| 1 | 2.8298 | 2.4613 |
| 2 | 1.4306 | 1.6169 |
| 3 | 1.0767 | 1.3434 |
| 4 | 0.8887 | 0.6507 |
| 5 | 0.7665 | 0.5088 |
| 6 | 0.6789 | 0.6881 |
| 7 | 0.6114 | 0.3687 |
| 8 | 0.5594 | 0.5343 |
| 9 | 0.5178 | 0.4041 |
| 10 | 0.4851 | 0.5481 |
