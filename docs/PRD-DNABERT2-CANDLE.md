# PRD: Treino real do DNABERT-2 no miner com Candle

## Problem Statement

O devnet actual do Xenomorph AI corre o miner em `--mock-mode` por defeito. O `model_id` é `multimolecule/dnabert2`, mas o miner não carrega nem treina os pesos reais do modelo. Para que o devnet seja um PoW útil de verdade, o miner deve ser capaz de:

1. Descarregar o modelo `multimolecule/dnabert2` e o respetivo tokenizer do Hugging Face.
2. Carregar os pesos `safetensors` numa implementação Rust nativa (Candle).
3. Executar um passo de treino Masked Language Modeling (MLM) num batch de sequências de DNA.
4. Submeter a prova de treino (`gradients_commitment`, `loss_before`, `loss_after`) para a seed-node.

## Solution

Criar um novo backend de treino `DnaBert2Trainer` no `xenom-miner` que implemente o trait `Trainer` existente. Este backend carrega o DNABERT-2 em memória, gera/usa batches de tokens MLM, corre um `forward + backward` com `candle`, e produz um `TrainingResult` compatível com o protocolo actual.

A seed-node continua a servir `TrainingBatch`es, mas o batch pode ser enriquecido com informação sobre o modelo e índices de dados. O próprio miner descarrega os ficheiros do Hugging Face localmente (`data_dir/models/<model_id>`) e reutiliza-os entre blocos.

## User Stories

1. Como miner do devnet, quero que o `xenom-miner` descarregue automaticamente o `multimolecule/dnabert2` na primeira corrida, para não ter de o fazer manualmente.
2. Como miner, quero escolher o backend de treino (`mock`, `cpu`, `dnabert2`) via CLI, para poder alternar entre validação rápida e treino real.
3. Como miner, quero que o `dnabert2` carregue os seus pesos do `model.safetensors`, para não depender de PyTorch.
4. Como miner, quero tokenizar sequências de DNA com o tokenizer BPE do DNABERT-2, para que as entradas sejam compatíveis com o modelo.
5. Como miner, quero que o batch de treino aplique máscaras aleatórias a 15% dos tokens (objectivo MLM), para reproduzir o pré-treino original.
6. Como miner, quero que o passo de treino calcule `loss_before` e `loss_after` e um `gradients_commitment` sobre os gradientes, para satisfazer o protocolo de prova.
7. Como operador de devnet, quero que o treino real seja desactivável ou váivel só em hardware capaz, para que o devnet não fique preso em CPU fraca.
8. Como operador, quero que o seed-node sirva o `base_checkpoint` (hash dos pesos) do modelo em vez de zeros, para que o miner possa verificar a linhagem do checkpoint.
9. Como tester, quero testes unitários para a arquitetura DNABERT-2, tokenizer e geração de batches MLM, para garantir que o treino produz saídas válidas.
10. Como tester, quero um teste de integração local que faça um forward+backward com uma versão miniaturada dos pesos, para validar o pipeline de treino sem descarregar 468 MB.

## Implementation Decisions

### 1. Upgrade da stack Candle

`xenom-miner` usa `candle-core` e `candle-nn` 0.3. Vamos migrar para a versão mais recente estável (ex. 0.8.x) e adicionar `candle-transformers` e `tokenizers` como dependências. Isto dá acesso a helpers de `VarBuilder`, `safetensors` e exemplos BERT que podem servir de base.

Risco: a API do `candle` mudou entre 0.3 e 0.8. O `CpuTrainer` e `MockTrainer` existentes podem precisar de ajustes menores.

### 2. Módulos novos no `xenom-miner`

- `src/model_hub.rs` — interface para descarregar `config.json`, `tokenizer.json` e `model.safetensors` do Hugging Face. Usa `reqwest` e guarda em `data_dir/models/<safe_model_id>`.
- `src/tokenizer.rs` — wrapper em torno do crate `tokenizers` para carregar `tokenizer.json` e fornecer `encode(text) -> Vec<u32>`.
- `src/models/dnabert2/` — implementação do `DnaBert2Model` e `DnaBert2ForMaskedLM` em Candle.
  - `config.rs` — parse do `config.json` do DNABERT-2.
  - `model.rs` — embeddings, encoder, ALiBi attention, pooler, MLM head.
  - `load.rs` — carregamento de pesos `safetensors` via `VarBuilder` com prefixos `dnabert2.*`.
- `src/data/dna_batch.rs` — gera sequências sintéticas de DNA (A,T,C,G), aplica máscaras MLM e cria `input_ids`, `labels`, `attention_mask`.
- `src/trainer/dnabert2.rs` — `DnaBert2Trainer` implementando `Trainer`.

### 3. Arquitetura DNABERT-2

Baseado no `multimolecule` source e no `config.json`:

- `DnaBert2Embeddings`: word embeddings + token type embeddings (sem positional embeddings; posições são codificadas via ALiBi na attention).
- `DnaBert2Encoder`: 12 camadas Transformer com `BertLayer` (self-attention + feed-forward). A self-attention inclui a bias ALiBi (linear attention bias).
- `DnaBert2Pooler`: simples camada densa + tanh sobre o token `[CLS]`.
- `DnaBert2LMHead`: transform + GELU + layer norm + decoder (vocab_size=4096).

ALiBi: em vez de position embeddings, a attention score recebe uma bias matrix `b * (i - j)` com slopes decrescentes por cabeça.

### 4. Training step

1. Descarregar/carregar modelo e tokenizer.
2. Criar batch: `batch_size` sequências de `seq_len` tokens, com 15% dos tokens mascarados (token 4 = `[MASK]`) e `labels` com os tokens originais.
3. Forward: `DnaBert2ForMaskedLM::forward(input_ids, token_type_ids, attention_mask)` -> logits `[batch, seq_len, vocab_size]`.
4. Calcular cross-entropy loss apenas sobre os tokens mascarados -> `loss_before`.
5. Backward: obter gradientes; aplicar um passo de optimizador SGD/AdamW -> `loss_after`.
6. Hash dos gradientes (flatten dos tensores de gradiente) com `blake3` -> `gradients_commitment`.
7. Preencher `TrainingResult` com `model_id`, `batch_indices`, `base_checkpoint`, `loss_before/after`, `gradients_commitment`, `compute_time_ms`.

### 5. Batch data flow

O `TrainingBatch` actual contém `base_checkpoint` (32 bytes) e `data_indices: Vec<u64>`. O `DnaBert2Trainer` usa `data_indices` como seed para gerar as sequências sintéticas de DNA desse batch. O `base_checkpoint` é o hash dos pesos do modelo; o miner pode comparar com o `weights_hash` local.

### 6. CLI e config

Adicionar argumento `--trainer` (ou manter `--mock-mode` como alias):

- `mock` — `MockTrainer` actual.
- `cpu` — `CpuTrainer` actual (MLP sintético).
- `dnabert2` — novo `DnaBert2Trainer`.

O default no devnet pode continuar a ser `mock` para não bloquear o devnet em CPU fraca. `--trainer=dnabert2` ativa treino real.

### 7. Seed-node checkpoint

Atualizar `seed-node/src/rpc/server.rs` para que `GetTrainingBatch` retorne o `base_checkpoint` real (`weights_hash` do `ModelCheckpoint`) quando o modelo estiver descarregado. Enquanto o download não termina, pode retornar zeros.

## Testing Decisions

- **Não testar implementação interna**, testar comportamento externo:
  - `DnaBert2Model` com entradas aleatórias produz `last_hidden_state` de shape `[batch, seq_len, hidden_size]`.
  - `DnaBert2ForMaskedLM` produz logits de shape `[batch, seq_len, vocab_size]`.
  - O tokenizer codifica `"ATCG"` numa sequência de ids e decodifica de volta.
  - `DnaBert2Trainer::train` reduz a loss (`loss_after < loss_before`) e produz um `gradients_commitment` não-zero.
- Prior art: testes em `xenom-miner/src/trainer/cpu_trainer.rs` e `xenom-miner/src/config.rs`.
- Testes de integração no crate `tests/` podem correr com um modelo pequeno gerado localmente (sem descarregar da net).

## Out of Scope

- Treino distribuído/multi-GPU.
- Datasets reais de DNA (GenBank/GUE). No devnet usam-se sequências sintéticas.
- Convergência/fine-tuning completo do DNABERT-2. O objectivo é um passo de treino por bloco, não treinar até convergência.
- Suporte a `pytorch_model.bin` (pickle). Usa-se `model.safetensors`.
- Conversão do modelo para ONNX/TorchScript.

## Further Notes

- O DNABERT-2 (`multimolecule/dnabert2`) é uma variação do MosaicBERT com ALiBi e tokenizer BPE de 4096 tokens. A implementação em Candle é funcionalmente equivalente ao Python `multimolecule` mas escrita em Rust.
- O `model.safetensors` tem ~468 MB. O devnet em CPU (2 vCPU / 8 GB) não será capaz de treinar este modelo em tempo real. O `--trainer=dnabert2` destina-se a hosts com GPU/CPU forte. Para validação do pipeline, sugere-se testar com um modelo menor (ex. `multimolecule/dnabert`) ou um dummy de teste.
- O download do modelo faz-se em background no `xenom-miner`; se não estiver disponível quando o primeiro batch chega, o miner pode fazer fallback para `mock` ou esperar (configurável).
