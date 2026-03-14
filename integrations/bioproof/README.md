# BioProof — Verified Scientific Computing on Xenom

BioProof is a **verifiable data integrity and scientific compute layer** built on top of the Xenom blockchain. It allows laboratories, research institutions and AI pipelines to:

- **Anchor** cryptographic proofs of datasets and pipeline outputs on the Xenom chain
- **Verify** that a file or result existed in a given state at a given point in time
- **Track lineage** between genomics artefacts (FASTQ → BAM → VCF → AI inference → report)
- **Coordinate** distributed scientific compute jobs and anchor proof-of-execution on-chain

> **BioProof does NOT store files on-chain.** Only compact cryptographic commitments (`proof_root`, `manifest_hash`) are anchored. All data stays off-chain.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      XENOM BLOCKCHAIN                   │
│                                                         │
│   tx.payload = AnchorPayload | JobAnchorPayload (JSON)  │
│   Provides: timestamp · immutability · public audit     │
└──────────────┬────────────────────────┬─────────────────┘
               │                        │
    ┌──────────▼──────────┐   ┌─────────▼──────────────┐
    │   bioproof-daemon   │   │   bioproof-worker       │
    │   (dataset anchor)  │   │   (compute job runner)  │
    └──────────┬──────────┘   └─────────┬──────────────┘
               │                        │
    ┌──────────▼────────────────────────▼──────────────┐
    │               bioproof-indexer                   │
    │         (scans chain, SQLite index)              │
    └──────────────────────┬───────────────────────────┘
                           │
    ┌──────────────────────▼───────────────────────────┐
    │               bioproof-api                       │
    │   REST: /anchor  /lineage  /verify  /health      │
    └──────────────────────────────────────────────────┘
```

### Three-layer separation

| Layer | What lives here |
|---|---|
| **Xenom chain** | Compact JSON anchors in `tx.payload` — timestamp, immutability, ordering |
| **BioProof service** | Chunking, hashing, signing, submission, indexing, verification |
| **Storage / compute** | Actual datasets, pipelines, model weights, AI outputs — off-chain |

---

## Crates

| Crate | Binary | Purpose |
|---|---|---|
| `bioproof-core` | — | Shared types, BLAKE3 hashing, secp256k1 signing |
| `bioproof-daemon` | `bioproof-daemon` | Anchor a file or dataset |
| `bioproof-indexer` | `bioproof-indexer` | Scan chain, index anchors in SQLite |
| `bioproof-verifier` | `bioproof-verifier` | Verify a certificate locally |
| `bioproof-api` | `bioproof-api` | REST API for queries and verification |
| `bioproof-worker` | `bioproof-worker` | Scientific Worker Node daemon |

---

## Quick Start

### Build all

```bash
cargo build -p bioproof-daemon -p bioproof-indexer \
            -p bioproof-verifier -p bioproof-api \
            -p bioproof-worker
```

---

## Use Case 1 — Dataset Anchoring

Anchor a VCF file to the Xenom chain with a verifiable certificate.

```bash
# 1. Anchor (dry-run — prints certificate JSON)
bioproof-daemon \
  --file       sample-001.vcf \
  --dataset-id sample-001 \
  --artifact-type vcf \
  --issuer     lab-genomics-x \
  --private-key <64-hex-privkey> \
  --out        sample-001.cert.json

# 2. Inspect certificate
cat sample-001.cert.json

# 3. Verify locally
bioproof-verifier \
  --cert sample-001.cert.json \
  --file sample-001.vcf

# 4. Anchor on-chain
bioproof-daemon \
  --file sample-001.vcf \
  --dataset-id sample-001 \
  --artifact-type vcf \
  --issuer lab-genomics-x \
  --private-key <64-hex-privkey> \
  --submit --node grpc://xenom-node:36669
```

### Certificate JSON structure

```json
{
  "manifest": {
    "dataset_id": "sample-001",
    "artifact_type": "vcf",
    "chunk_size": 4194304,
    "file_hash": "blake3-hex...",
    "proof_root": "merkle-root-hex...",
    "pipeline_hash": null,
    "model_hash": null,
    "parent_root": "blake3-of-parent-vcf...",
    "issuer": "lab-genomics-x",
    "created_at": 1773390000
  },
  "manifest_hash": "blake3-of-manifest-json...",
  "issuer_sig": "der-sig-hex...",
  "issuer_pubkey": "compressed-pubkey-hex...",
  "txid": "xenom-txid-hex...",
  "daa_score": 12345678,
  "anchored_at": null
}
```

### Lineage tracking

Use `parent_root` to link artefacts:

```
FASTQ (parent_root=null)
  └── BAM  (parent_root=fastq.proof_root)
        └── VCF  (parent_root=bam.proof_root)
              └── AI output (parent_root=vcf.proof_root)
                       └── Report (parent_root=ai.proof_root)
```

---

## Use Case 2 — Scientific Worker Node

A worker node picks up compute jobs, executes the pipeline, and anchors the result.

### 1. Start the worker daemon

```bash
bioproof-worker \
  --private-key <64-hex-privkey> \
  --inbox      ./job-inbox \
  --work-root  ./work \
  --node       grpc://xenom-node:36669 \
  --submit
```

On startup, the worker:
1. Detects hardware (CPU/RAM/GPU via `sysinfo` + `nvidia-smi`/`rocm-smi`)
2. Detects installed software (`docker`, `singularity`, `nextflow`, `snakemake`, `python3`, etc.)
3. Writes `work/capabilities.json`
4. Polls `job-inbox/` every 3 seconds for `*.json` job files

### 2. Submit a job

Drop a `ComputeJob` JSON file into `job-inbox/`:

```json
{
  "job_id": "var-calling-001",
  "job_type": "variant_calling",
  "input_root": "blake3-merkle-root-of-input-dir...",
  "pipeline_hash": "blake3-of-pipeline-script...",
  "container_hash": "docker.io/broadinstitute/gatk:4.5.0.0",
  "model_hash": null,
  "reward_sompi": 50000000,
  "max_time_secs": 3600,
  "requester_pubkey": "compressed-pubkey-hex...",
  "posted_at": 1773390000
}
```

Prepare the working directory:

```
work/
└── var-calling-001/
    ├── input/          ← put input files here
    └── pipeline        ← the pipeline script (bash/python/nf/Snakefile)
```

### 3. What the worker does

```
1. Verifies input_root matches actual input files (tamper detection)
2. Picks best executor:
     Nextflow > Docker/Singularity > AI (PyTorch) > Native (bash/python)
3. Runs pipeline with:
     INPUT_DIR  = work/<job_id>/input/
     OUTPUT_DIR = work/<job_id>/output/
4. Hashes each output file with BLAKE3 (chunk + Merkle per file)
5. Computes combined output_root (Merkle over all output proof_roots)
6. Builds ComputeJobManifest including execution_trace_hash
7. Signs manifest with worker secp256k1 key
8. Emits JobAnchorPayload → anchors in Xenom tx.payload
9. Moves job file to job-inbox/done/
```

### Execution backends

| Backend | Trigger | Requires |
|---|---|---|
| **Nextflow** | `.nf` file or nextflow installed | `nextflow` in PATH |
| **Snakemake** | snakemake installed | `snakemake` in PATH |
| **Docker** | `container_hash` field set | `docker info` succeeds |
| **Singularity** | Singularity/Apptainer installed | HPC-compatible rootless |
| **AI (PyTorch)** | `python3 -c 'import torch'` succeeds | CUDA/ROCm optional |
| **Native** | Always available (fallback) | `bash` / `python3` |

### Worker capabilities JSON

```json
{
  "worker_pubkey": "compressed-pubkey-hex...",
  "hostname": "node-01",
  "cpu_count": 32,
  "ram_mib": 131072,
  "disk_mib": 2048000,
  "gpus": [
    { "index": 0, "model": "NVIDIA A100", "vram_mb": 81920, "cuda": true, "rocm": false }
  ],
  "backends": ["docker", "nextflow", "native", "cuda"],
  "job_types": ["genomics_pipeline", "variant_calling", "ai_inference", "ai_training"],
  "max_concurrency": 4,
  "measured_at": 1773390000
}
```

---

## Use Case 3 — Indexer + API

### Start the indexer

```bash
bioproof-indexer \
  --node    grpc://xenom-node:36669 \
  --db-path bioproof.db \
  --poll-ms 2000
```

The indexer polls `get_blocks`, parses `tx.payload` for BioProof anchors (`app = "bioproof"` or `"bioproof-job"`) and stores them in SQLite.

### Start the API

```bash
bioproof-api \
  --db-path bioproof.db \
  --listen  0.0.0.0:8090
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service liveness |
| `GET` | `/api/anchor/:proof_root` | Fetch anchor by proof_root |
| `GET` | `/api/anchors?kind=vcf&limit=50` | List anchors with filter |
| `GET` | `/api/lineage/:proof_root` | Resolve on-chain lineage |
| `POST` | `/api/verify` | Verify certificate (manifest_hash + sig + optional on-chain) |

### Verify via API

```bash
curl -X POST http://localhost:8090/api/verify \
  -H 'Content-Type: application/json' \
  -d '{"cert": <certificate-json>, "manifest_only": false}'
```

Response:

```json
{
  "manifest_hash_ok": true,
  "signature_ok": true,
  "proof_root_ok": true,
  "overall": true,
  "errors": []
}
```

---

## On-chain data format

BioProof uses the native Kaspa `tx.payload` field (not OP_RETURN).

### Dataset anchor (`app = "bioproof"`)

```json
{
  "app": "bioproof",
  "v": 1,
  "proof_root": "hex...",
  "manifest_hash": "hex...",
  "kind": "vcf"
}
```

### Compute job result (`app = "bioproof-job"`)

```json
{
  "app": "bioproof-job",
  "v": 1,
  "job_id": "var-calling-001",
  "manifest_hash": "hex...",
  "output_root": "hex...",
  "worker_pubkey": "hex...",
  "worker_sig": "der-hex..."
}
```

---

## Supported job types

| JobType | Description |
|---|---|
| `genomics_pipeline` | General Nextflow/Snakemake genomics workflow |
| `variant_calling` | GATK, DeepVariant, freebayes |
| `genome_assembly` | SPAdes, Flye, Hifiasm |
| `single_cell_rna` | Seurat, Scanpy, Cell Ranger |
| `metagenomics` | Kraken2, MetaPhlAn, DIAMOND |
| `ai_inference` | PyTorch/TF model inference |
| `ai_training` | GPU model fine-tuning |
| `protein_folding` | AlphaFold2-style prediction |
| `molecular_dynamics` | GROMACS, AMBER, OpenMM |

---

## Security model

- All manifests are signed with **secp256k1** (same curve as Xenom wallet keys)
- `manifest_hash` = BLAKE3 of canonical manifest JSON
- `proof_root` = BLAKE3 binary Merkle tree root over fixed-size chunks
- Worker signs `ComputeJobManifest` → `worker_sig` goes on-chain
- Anyone can re-hash the original file and verify against the on-chain `proof_root`
- The Xenom chain provides **public ordering** — no trusted timestamp authority needed

---

## Pending / not yet implemented

- **Transaction submission** (`bioproof-daemon --submit`, `bioproof-worker --submit`): UTXO fetch + tx building + signing. Marked as `TODO` in `submit_anchor` / `submit_job_anchor`.
- **Job registry**: on-chain posting of `ComputeJob` offers and worker bid/claim flow.
- **Reward settlement**: XEN payment from requester to worker after verified completion.
- **Partial recomputation verification**: challenger nodes re-run a deterministic subset to slash bad workers.
