# Genetics L2 — Scientific Compute Layer on Xenom

Genetics L2 is a **Layer-2 compute network** built on top of Xenom's Layer-1 blockchain. It enables distributed scientific computation (genomics, proteomics, AI inference) using miners as compute workers, with results anchored immutably on-chain.

> **Layer-1 is untouched.** No consensus changes. L2 lives entirely in `integrations/genetics-l2/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   EXTERNAL SCIENTIFIC SOURCES                   │
│  Kaggle · NIH/NCBI · NIH Challenge · DREAM · Horizon Prize      │
└───────────────────────────┬─────────────────────────────────────┘
                            │  job-fetcher polls APIs
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              genetics-l2-coordinator  :8091                     │
│                                                                 │
│  SQLite: jobs · results · validations · payouts                 │
│  REST API: /jobs  /results  /validations  /payouts  /stats      │
└─────┬──────────────────────────┬──────────────────┬────────────┘
      │                          │                  │
      ▼                          ▼                  ▼
genetics-l2-worker       genetics-l2-validator  genetics-l2-settlement
(miners run algorithms)  (partial recompute)    (results_root → Xenom tx)
      │                          │                  │
      └──────────────────────────┘                  │
                                                    ▼
                                      ┌─────────────────────────┐
                                      │    XENOM BLOCKCHAIN      │
                                      │  tx.payload = settlement │
                                      └─────────────────────────┘
```

### Full data flow

```
external sources (Kaggle / NIH / BOINC)
         │
   job-fetcher polls every N minutes
         │  POST /jobs
         ▼
   coordinator stores job in SQLite (status = open)
         │
   worker polls GET /jobs?status=open
         │  POST /jobs/:id/claim  → status = claimed
         ▼
   worker downloads dataset, runs algorithm
         │  POST /results  → status = completed
         ▼
   validator polls GET /jobs?status=completed
         │  partial recomputation → ValidationReport
         │  POST /validations  → status = validated
         ▼
   settlement polls GET /jobs?status=validated
         │  builds results_root (Merkle over all valid result_roots)
         │  POST /payouts (winner worker)
         │  anchors SettlementPayload in Xenom tx.payload
         ▼
   job status = settled  ✓
```

---

## Crates

| Crate | Binary | Role |
|---|---|---|
| `genetics-l2-core` | — | Shared types: `ScientificJob`, `JobResult`, `ValidationReport`, `SettlementPayload` |
| `genetics-l2-coordinator` | `genetics-l2-coordinator` | REST API, SQLite job registry, payout tracking |
| `genetics-l2-fetcher` | `genetics-l2-fetcher` | Polls Kaggle, NIH/NCBI, NIH Challenge, DREAM, Horizon; registers jobs |
| `genetics-l2-worker` | `genetics-l2-worker` | Claims + executes jobs, submits results |
| `genetics-l2-validator` | `genetics-l2-validator` | Partial recomputation, hash verification |
| `genetics-l2-settlement` | `genetics-l2-settlement` | Creates `results_root`, pays winner, anchors on Xenom |

---

## Build

```bash
cargo build \
  -p genetics-l2-coordinator \
  -p genetics-l2-fetcher \
  -p genetics-l2-worker \
  -p genetics-l2-validator \
  -p genetics-l2-settlement
```

---

## Devnet Quick Start (all-in-one)

```bash
export BIN="/path/to/Xenomorph/target/release"
export PRIVKEY="<64-hex-secp256k1-key>"
export MINING_ADDR="xenomdev:<your-devnet-address>"
export COORDINATOR="http://localhost:8091"

# T1 — Node
$BIN/xenom --devnet --utxoindex

# T2 — Coordinator
$BIN/genetics-l2-coordinator --db-path /tmp/genetics-l2.db --listen 0.0.0.0:8091

# T3 — Stratum bridge
$BIN/xenom-stratum-bridge --mining-address "$MINING_ADDR" --rpcserver 127.0.0.1:18610 --listen 0.0.0.0:5555 --l2-coordinator "$COORDINATOR" --l2-theme genetics --devnet

# T4 — Fetcher (pick one or more sources)
$BIN/genetics-l2-fetcher --coordinator "$COORDINATOR" --nih-challenges --horizon --poll-secs 300

# T5 — GPU Miner + L2 worker (NIH/Horizon — no perch_infer)
$BIN/genome-miner gpu --devnet --mining-address "$MINING_ADDR" --stratum stratum+tcp://127.0.0.1:5555 --gpu 0 --l2-coordinator "$COORDINATOR" --l2-private-key "$PRIVKEY"

# T6 — Validator
$BIN/genetics-l2-validator --private-key "$PRIVKEY" --coordinator "$COORDINATOR"

# T7 — Settlement (--devnet for xenomdev: address prefix)
$BIN/genetics-l2-settlement --coordinator "$COORDINATOR" --node grpc://localhost:18610 --private-key "$PRIVKEY" --devnet --submit --poll-ms 15000
```

> **Funding the settlement wallet:** the settlement daemon signs anchor transactions from the `xenomdev:` address derived from `$PRIVKEY`. Mine a few blocks first so that address receives coinbase rewards, then the daemon will anchor automatically.

---

## Detailed Setup

### 1. Start the coordinator

```bash
genetics-l2-coordinator \
  --db-path genetics-l2.db \
  --listen  0.0.0.0:8091
```

### 2. Start the fetcher (Kaggle + NIH)

```bash
genetics-l2-fetcher \
  --coordinator http://localhost:8091 \
  --kaggle-key  <username>:<api-token> \
  --poll-secs   300
```

The fetcher automatically discovers genomics competitions on Kaggle and public SRA samples on NCBI, and registers them as jobs.

### 3. Start workers (miners)

```bash
genetics-l2-worker \
  --private-key <64-hex-privkey> \
  --coordinator http://localhost:8091 \
  --work-root   ./work \
  --poll-ms     5000
```

Multiple workers can run simultaneously; the coordinator uses optimistic concurrency — the first claim wins.

### 4. Start the validator

```bash
genetics-l2-validator \
  --private-key    <64-hex-privkey> \
  --coordinator    http://localhost:8091 \
  --score-tolerance 0.05
```

### 5. Start the settlement service

```bash
# Dry-run (default — inspect anchors without submitting)
genetics-l2-settlement \
  --coordinator http://localhost:8091 \
  --node        grpc://localhost:36669

# Submit to chain (mainnet)
genetics-l2-settlement \
  --coordinator http://localhost:8091 \
  --node        grpc://localhost:36669 \
  --private-key <HEX> \
  --submit

# Submit to chain (devnet — uses xenomdev: address prefix)
genetics-l2-settlement \
  --coordinator http://localhost:8091 \
  --node        grpc://localhost:18610 \
  --private-key <HEX> \
  --devnet --submit
```

---

## REST API Reference

### Coordinator  `http://localhost:8091`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service liveness |
| `GET` | `/stats` | Job + result + payout counters |
| `GET` | `/jobs?status=open&limit=50` | List jobs by status |
| `POST` | `/jobs` | Register a new job (used by fetcher) |
| `POST` | `/jobs/:id/claim` | Worker claims a job |
| `POST` | `/results` | Worker submits a result |
| `GET` | `/results/:job_id` | All results for a job (sorted by score) |
| `POST` | `/validations` | Validator posts a validation report |
| `GET` | `/payouts?worker=<pubkey>` | List payouts |
| `POST` | `/payouts` | Settlement registers a payout |

### Job status lifecycle

```
open → claimed → completed → validated → settled
                          └→ failed (validator: invalid)
```

---

## Data Types

### ScientificJob (posted by fetcher)

```json
{
  "job_id": "kaggle-variant_calling-3f2a",
  "source": "kaggle",
  "external_ref": "genomics-bowel-disease",
  "dataset_root": "blake3-hex...",
  "dataset_url": "https://www.kaggle.com/c/genomics-bowel-disease/data",
  "algorithm": "variant_calling",
  "task_description": "Genomics of Bowel Disease — variant calling challenge",
  "reward_sompi": 50000000000,
  "max_time_secs": 86400,
  "status": "open",
  "claimed_by": null,
  "created_at": 1773390000
}
```

### JobResult (submitted by worker)

```json
{
  "result_id": "kaggle-variant_calling-3f2a-a1b2c3d4",
  "job_id": "kaggle-variant_calling-3f2a",
  "worker_pubkey": "compressed-secp256k1-hex...",
  "result_root": "blake3-merkle-root-of-output-files...",
  "score": 1823.47,
  "trace_hash": "blake3-of-stdout-stderr...",
  "worker_sig": "secp256k1-der-hex...",
  "submitted_at": 1773393600
}
```

### ValidationReport (produced by validator)

```json
{
  "report_id": "kaggle-variant_calling-3f2a-a1b2c3d4-val-3f2b",
  "job_id": "kaggle-variant_calling-3f2a",
  "result_id": "kaggle-variant_calling-3f2a-a1b2c3d4",
  "validator_pubkey": "compressed-secp256k1-hex...",
  "verdict": "valid",
  "recomputed_score": 1821.19,
  "score_delta": 0.00125,
  "notes": "score within tolerance (0.0012 < 0.0500)",
  "validator_sig": "secp256k1-der-hex...",
  "validated_at": 1773394200
}
```

### SettlementPayload (anchored on Xenom `tx.payload`)

```json
{
  "app": "genetics-l2",
  "v": 1,
  "job_id": "kaggle-variant_calling-3f2a",
  "source": "kaggle",
  "algorithm": "variant_calling",
  "dataset_root": "blake3-hex...",
  "results_root": "blake3-merkle-root-over-all-valid-results...",
  "best_score": 1823.47,
  "winner_pubkey": "compressed-secp256k1-hex...",
  "settled_at": 1773394800
}
```

---

## Supported Algorithms

| Algorithm | Description | Typical inputs | Typical outputs |
|---|---|---|---|
| `sequence_alignment` | Generic pairwise/MSA | FASTQ, FASTA | SAM, BAM |
| `smith_waterman` | Local alignment (SIMD) | FASTA sequences | Alignment score, aligned pairs |
| `needleman_wunsch` | Global alignment | FASTA sequences | Alignment score, aligned pairs |
| `variant_calling` | SNP/INDEL detection | BAM + reference | VCF |
| `genome_assembly` | De novo assembly | Reads (short/long) | Contigs FASTA |
| `protein_folding` | Structure prediction | Protein FASTA | PDB |
| `rna_expression` | Differential expression | FASTQ + annotation | Count matrix TSV |
| `metagenomics` | Taxonomy classification | Shotgun reads | Kraken2 report |
| `molecular_docking` | Ligand binding | PDB + ligand SDF | Docking score |
| `drug_discovery` | Virtual screening / ADMET | SMILES, PDB | Docking score, toxicity |
| `cancer_genomics` | Somatic mutation, CNV, fusion | BAM + VCF | Annotated variants |
| `biomarker_discovery` | Omics biomarker identification | Expression matrix | Biomarker list |
| `network_biology` | Gene regulatory network inference | Expression data | Network GML |
| `gene_expression` | Bulk / single-cell prediction | FASTQ / h5ad | Count matrix |
| `digital_health` | Health data analytics / e-health | HL7, FHIR, CSV | Clinical insights |
| `biotechnology` | Synthetic biology, cell engineering | Sequence files | Engineered constructs |

---

## External Sources

### Kaggle  (`--kaggle-key username:token`)

- Lists genomics / biology competitions via Kaggle API v1
- Filters by tags: `genomics`, `biology`, `dna`, `protein`, `rna`, `genetics`, etc.
- Infers algorithm from competition title/tags
- Converts USD prize to sompi reward (approximate)

Get your key at: https://www.kaggle.com/account → API → Create New Token

### NIH / NCBI  (no key — `--nih` enabled by default)

- Queries NCBI E-utilities SRA database for recent variant-calling datasets
- Uses public REST API (100 requests/minute without key)
- Generates one job per SRA accession

### NIH Prize Challenges  (`--nih-challenges`)

- Queries [challenge.gov](https://api.challenge.gov/api/3/action/package_search?q=NIH) CKAN API
- Filters results by NIH/HHS/biomedical keywords
- Maps each open challenge to a `ScientificJob` with appropriate algorithm

### DREAM Challenges  (`--dream [--synapse-pat TOKEN]`)

- Queries [Synapse API](https://repo-prod.prod.sagebase.org/repo/v1/) for active DREAM challenges
- Supports authenticated access via `--synapse-pat` or `SYNAPSE_PAT` env var
- Generates sub-tasks per challenge phase

### EU Horizon Prizes  (`--horizon`)

- Queries [EuropePMC REST API](https://europepmc.org/RestfulWebService) (EBI-hosted, no auth)
- Returns EC-funded biomedical research in genomics / health / biotechnology
- Maps to `HorizonPrize` source with €1M–€5M reward bracket signals
- Prize range: **€1M – €5M** per challenge

### BOINC  (`--boinc-url http://project-server/`)

- Reads project info from BOINC project server XML endpoint
- Generates compute jobs for volunteer science projects
- Supports any BOINC project that exposes standard endpoints

---

## Security Model

- **Claim races** are handled via optimistic locking in SQLite — only one worker succeeds
- **Score fraud prevention** — validators perform independent partial recomputation with a configurable tolerance (default 5%)
- **Multiple validators** can submit reports for the same result — majority verdict applies
- **Result hashing** — `result_root` is a BLAKE3 Merkle root over all output files, making tampering detectable
- **Worker identity** — each result is signed with the worker's secp256k1 private key
- **Settlement is on-chain** — `SettlementPayload` is stored in Xenom `tx.payload`, providing immutable public proof of who won and what the result was

---

## Adding a New External Source

Implement the `SourceFetcher` trait in `genetics-l2-fetcher/src/<name>.rs`:

```rust
pub struct MyFetcher { ... }

#[async_trait::async_trait]
impl SourceFetcher for MyFetcher {
    fn name(&self) -> &str { "my-source" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // Query your API, return ScientificJob list
    }
}
```

Then add it to the `fetchers` vec in `main.rs`.

---

## Adding a New Algorithm

Add a variant to the `Algorithm` enum in `genetics-l2-core/src/lib.rs`, then handle it in `genetics-l2-worker/src/main.rs`:

```rust
Algorithm::MyAlgorithm => {
    my_algorithm_impl(&input_dir, &output_dir).await
}
```

The function returns `(f64, String)` — score and execution trace.

---

## Relation to BioProof

Genetics L2 and BioProof are complementary:

| BioProof | Genetics L2 |
|---|---|
| Anchors **datasets and pipeline outputs** | Anchors **compute job settlements** |
| Identity: lab / researcher | Identity: miner / worker |
| Any file type | Genomics / proteomics algorithms |
| Manual submission | Fully automated daemon |
| `app = "bioproof"` on-chain | `app = "genetics-l2"` on-chain |

Both use the same Xenom `tx.payload` mechanism and BLAKE3 + secp256k1 security primitives from `bioproof-core`.

---

## Pending

- **Anchor UTXO funding**: settlement wallet must hold mature UTXOs to pay anchor tx fees; add faucet or auto-fund step for devnet CI
- **Multi-validator consensus**: aggregate reports from N validators before marking as validated
- **Reward escrow**: lock `reward_sompi` at job posting time so workers are guaranteed payment
- **climate-l2**: same architecture applied to climate modelling (CMIP, ERA5, ECMWF)
- **DREAM ensemble scoring**: aggregate miner results before validator, per DREAM challenge rules
