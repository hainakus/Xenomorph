# mini-genome-model (MGM-1 v2)

Small masked-language-model transformer for genomic DNA sequences, built on
[Candle](https://github.com/huggingface/candle).  MGM-1 v2 adds several
anti-overfitting changes while keeping the model fast enough to train on a
single CPU or small GPU.

## Architecture

- **Vocabulary**: 8 tokens (A, C, G, T, `[MASK]`, padding, `[CLS]`, `[SEP]`).
- **Token + sinusoidal positional embeddings**.
- **Stacked Transformer blocks** with pre-layer-normalization, multi-head
  self-attention and a ReLU feed-forward network.
- **Shared classification head** over the first 4 logits for MLM (A/C/G/T).
- Typical default size: `d_model=128`, `n_layers=4`, `n_heads=4`, `d_ff=512`,
  `max_seq_len=512` (~1–2 M parameters).

## Anti-overfitting techniques (MGM-1 v2)

| Technique | Default v2 | Effect |
| --- | --- | --- |
| Dropout | `0.2` (was `0.1`) | Randomly zeroes attention/FFN activations during training. |
| Label smoothing | `0.1` | Softens the MLM target distribution to prevent overconfident / collapsed predictions. |
| Weight decay | `0.01` (AdamW) | Decoupled L2 regularization on all trainable weights. |
| Gradient clipping | global norm `1.0` | Caps the L2 norm of the gradient vector, preventing a single noisy batch from blowing up. |
| Class weights | `[1.5, 1.5, 1.5, 1.0]` for A/C/G/T | Up-weights under-represented bases in the MLM loss. |

Class weights are applied **to the loss**, not to the logits before softmax.

## Usage

```rust
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use mini_genome_model::{MiniGenomeConfig, MiniGenomeModel, MiniGenomeTrainer};

let device = Device::Cpu;
let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

let config = MiniGenomeConfig::default();
let trainer = MiniGenomeTrainer::new(vb, config, &device).unwrap();

// input_ids and labels are (batch, seq_len) i64 tensors; 4 = [MASK]
let (loss, accuracy, grad_norm) = trainer.train_step(&input_ids, &labels, 1e-3).unwrap();
```

## Crate layout

- `src/lib.rs` — `MiniGenomeConfig`, `MiniGenomeModel`, `MiniGenomeTrainer`,
  tokenizer, and unit tests.
