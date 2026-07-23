# Estado da Implementação — branch `xenom-AI-v2`

Relatório gerado a partir do estado actual do repositório e da sessão de desenvolvimento.

---

## Commits recentes

```text
aa8419c Add bounded-staleness checkpoint cache to seed-node
b26b735 Speed up seed-node FedAvg by avoiding temp file and extra weight reads
1babf08 Add MultiGpuTrainer phase timing logs
0e94b82 Avoid CPU round-trip when averaging multi-GPU gradients
e67ef66 Add gradient compression CLI and 10-BPS simnet stress script
5ecff33 Fix remote miner WebSocket timeouts on gradient submission
421ae48 Reject FedAvg gradients for stale checkpoints
e5182fd Add weighted FedAvg and top-k gradient compression
06bab2d Keep persistent AdamW state in FedAvg aggregator
39a84d8 Wire FedAvg gradient aggregation end-to-end (#60)
```

Working tree: **limpa** (nenhuma alteração por commitar).

---

## Funcionalidades implementadas nesta sessão

### 1. Média de gradientes multi-GPU na GPU master

- **Ficheiro:** `xenom-miner/src/trainer/multi_gpu.rs`
- **Alteração:** os gradientes deixaram de ser copiados para a CPU; são movidos para a GPU master e aí são média/accumulação.
- **Impacto:** elimina o gargalo de round-trip CPU-GPU durante o treino.

### 2. Logs de timing no `MultiGpuTrainer`

- **Ficheiro:** `xenom-miner/src/trainer/multi_gpu.rs`
- **Alteração:** emissão de `INFO` por bloco com breakdown do tempo em:
  - `compute`, `gather`, `overflow_check`, `avg`, `add`, `apply`, `loss_after`, `commitment`
  - e `Gradient update build time`.
- **Impacto:** permite identificar onde o tempo está a ser gasto (CPU vs GPU, treino vs overhead).

### 3. Otimização do `submit_gradients` no seed-node

- **Ficheiro:** `seed-node/src/model/manager.rs`
- **Alterações:**
  - Adicionado `DnaBert2Trainer::save_weights_to_bytes()` para serialização em memória.
  - Adicionado `ModelStorage::load_model_metadata()` para ler só `config.enc` e `tokenizer.enc`.
  - FedAvg deixa de escrever ficheiro `safetensors` temporário e de reler o `weights.enc` antigo.
- **Impacto:** reduz I/O de disco na agregação, acelerando a produção de novos checkpoints.

### 4. Cache de checkpoints com bounded staleness

- **Ficheiro:** `seed-node/src/model/manager.rs`
- **Configuração:**
  - `XENO_CHECKPOINT_HISTORY_SIZE=8` (default)
  - `FEDAVG_MIN_PARTICIPANTS=4` (default devnet)
- **Comportamento:**
  - O seed-node guarda em memória até 8 lineages de checkpoints.
  - Aceita gradientes cujo `base_checkpoint` esteja dentro dessa janela.
  - Cada cache entry tem o seu próprio `DnaBert2Trainer` (CPU) e `FedAvgAggregator`.
  - Quando o número de participantes por base atinge `FEDAVG_MIN_PARTICIPANTS`, aplica a média e avança o `head_hash` dessa lineage.
  - Só promove o novo hash a `active_checkpoint` se o `head_hash` anterior era o ativo.
  - Insere novo key para o novo head, permitindo que a lineage continue.
- **Impacto:** o miner já não precisa de esperar pela agregação a cada bloco; pode submeter vários gradientes sobre o mesmo `base_checkpoint` antes de receber novo checkpoint.

### 5. Configuração do devnet

- **Ficheiros:** `.env.example`, `scripts/run-native-devnet.sh`
- **Adicionado:**
  - `XENO_CHECKPOINT_HISTORY_SIZE=8`
  - `FEDAVG_MIN_PARTICIPANTS=4`

---

## Resultados de testes

```bash
cargo test -p seed-node -p xenom-miner
```

- `seed-node`: 41 tests passed
- `xenom-miner`: 34 tests passed
- `integration_tests`: 2 tests passed

`cargo fmt --all -- --check`: OK  
`cargo clippy -p seed-node`: sem novos warnings no `manager.rs` (warnings antigos noutros ficheiros persistem).

---

## Estado da blockchain neste branch

| Aspeto | Estado |
|---|---|
| **Dificuldade / DAA** | Não alterado. Devnet mantém `target_time_per_block=1000 ms` e `max_difficulty_target=2^255-1`. Com 1× 3080 a produzir muito abaixo de 1 BPS, a dificuldade fica no piso mínimo. |
| **Produção de blocos** | Agora limitada sobretudo por `treino + agregação`. Com `FEDAVG_MIN_PARTICIPANTS=4`, o miner submete 4 gradientes no mesmo `base_checkpoint` antes de esperar por um novo active checkpoint. |
| **Utilização GPU** | Ainda não é contínua. Os logs de `nvidia-smi dmon` mostram picos de 38-85% `sm` seguidos de longos períodos a 0%. Para atingir ocupação elevada, é necessário: (a) aumentar `--genome-batch-size` e `--gradient-accumulation` para o treino durar mais; (b) usar `--gradient-top-k-ratio 0.1` para reduzir o tempo de agregação. |
| **Convergência / FedAvg** | Suporta múltiplas submissões por base e stale gradients dentro da cache. A lineage ativa avança correctamente; gradientes fora da cache são rejeitados. |
| **Finalidade** | `finality_depth=86400` blocos. Para devnet, isto continua alto se o block time subir. |

---

## Recomendação de execução

Para testar ocupação GPU elevada e fluxo contínuo:

```bash
# seed-node / xeno-node
FEDAVG_MIN_PARTICIPANTS=4 XENO_CHECKPOINT_HISTORY_SIZE=8 ./xenom ...

# miner
./target/release/xenom-miner --trainer dnabert2 \
  --rpc-url ws://127.0.0.1:17110 \
  --gpus 0,1,2 \
  --genome-batch-size 64 \
  --gradient-accumulation 3 \
  --gradient-top-k-ratio 0.1
```

Estimativa esperada com treino de ~30 s e agregação de ~10 s:

```text
tempo de round = 4 × 30 s + 10 s = 130 s
tempo GPU ativa = 4 × 30 s = 120 s
utilização GPU ≈ 92%
```

---

## Notas

- O override `XENO_TARGET_BLOCK_TIME_MS` não está neste branch; foi implementado e posteriormente perdido durante a sessão. Se for necessário tornar a dificuldade sensível a 1× 3080, terá de ser reintroduzido.
- A utilização da CPU no seed-node continua a ser o factor limitante quando `--gradient-top-k-ratio` é 1.0 (payload denso de 440 MB).
