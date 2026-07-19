# Issue 4: DnaBert2 model forward pass

## Parent

PRD: `docs/PRD-DNABERT2-CANDLE.md`

## What to build

Implement the DNABERT-2 model architecture in Candle. The model is a BERT variant with no learned positional embeddings and ALiBi attention biases. Build `DnaBert2Embeddings`, `DnaBert2Encoder` (with `DnaBert2Layer`, `DnaBert2SelfAttention`, `DnaBert2Intermediate`, `DnaBert2Output`), `DnaBert2Pooler`, and `DnaBert2ForMaskedLM`. The forward pass should accept `input_ids`, `token_type_ids`, and an optional `attention_mask`, and return logits of shape `[batch, seq_len, vocab_size]`. Implement ALiBi as a fixed linear bias applied to attention scores, with slopes decreasing per head.

## Acceptance criteria

- [ ] `DnaBert2Model` produces `last_hidden_state` of shape `[batch, seq_len, hidden_size]`.
- [ ] `DnaBert2ForMaskedLM` produces `logits` of shape `[batch, seq_len, vocab_size]`.
- [ ] A forward pass with random input completes without panics.
- [ ] Unit tests assert output shapes and that the model uses the loaded weights.

## Blocked by

- `docs/issues/03-config-safetensors-loading.md`
