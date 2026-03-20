# Genetics L2 Coordinator — REST API

Base URL: `http://localhost:8091`

---

## Health & Info

### `GET /health`
Returns `200 OK` if the service is running.

```json
{ "status": "ok" }
```

### `GET /pubkey`
Returns the coordinator's secp256k1 public key (used by miners to encrypt result payloads).

```json
{ "pubkey": "02a1b2c3..." }
```

### `GET /stats`
Returns job and result counters.

```json
{
  "total_jobs": 12,
  "open_jobs": 2,
  "completed_jobs": 4,
  "validated_jobs": 3,
  "settled_jobs": 3,
  "total_results": 8,
  "total_payouts": 3
}
```

---

## Jobs

### `GET /jobs`
List jobs with optional filters.

| Query param | Type   | Default | Description                                |
|-------------|--------|---------|--------------------------------------------|
| `status`    | string | `open`  | `open` / `claimed` / `completed` / `validated` / `settled` |
| `limit`     | int    | `50`    | Max results                                |
| `offset`    | int    | `0`     | Pagination offset                          |

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "kaggle-acoustic_classification-b322",
      "source": "kaggle",
      "external_ref": "birdclef-2026",
      "dataset_root": "c71daa77...",
      "dataset_url": "kaggle://competitions/birdclef-2026",
      "algorithm": "acoustic_classification",
      "task_description": "BirdCLEF+ 2026 — Acoustic species identification",
      "reward_sompi": 100000000000,
      "max_time_secs": 86400,
      "status": "open",
      "claimed_by": null,
      "created_at": 1773712162
    }
  ]
}
```

### `POST /jobs`
Create a new job manually.

**Body:** `ScientificJob` JSON (same fields as above, minus auto-generated ones).

### `POST /jobs/:job_id/claim`
Miner claims a job before executing it.

**Body:**
```json
{ "worker_pubkey": "02abcd..." }
```

**Response:** `200` if claimed, `409` if already taken.

---

## Results

### `POST /results`
Miner submits a completed result. Marks job as `completed`.

**Body:**
```json
{
  "result_id": "kaggle-acoustic_classification-b322-a1b2c3d4",
  "job_id": "kaggle-acoustic_classification-b322",
  "worker_pubkey": "02abcd...",
  "result_root": "",
  "score": 0.0,
  "trace_hash": null,
  "worker_sig": "3045...",
  "encrypted_payload": "0a1b2c...",
  "ephemeral_pubkey": "03ef...",
  "submitted_at": 1773712500
}
```

> `result_root`, `score`, and `trace_hash` are zeroed after encryption.  
> `encrypted_payload` contains the real values + `predictions_csv` encrypted with the coordinator's public key (ECIES).

### `GET /results/:job_id`
Returns all results for a job. The `score` field returns `recomputed_score` from the validator when available.

```json
{
  "job_id": "kaggle-acoustic_classification-b322",
  "results": [
    {
      "result_id": "...",
      "worker_pubkey": "02abcd...",
      "result_root": "...",
      "score": 0.85,
      "submitted_score": 0.0,
      "recomputed_score": 0.85,
      "verdict": "valid",
      "submitted_at": 1773712500,
      "encrypted_payload": "0a1b2c...",
      "ephemeral_pubkey": "03ef..."
    }
  ]
}
```

### `GET /results/:job_id/csv`
Coordinator decrypts the best valid result's `encrypted_payload` and returns `predictions_csv` as `text/csv`.

```
filename,confidence
XC134896.ogg,0.823400
XC201234.ogg,0.712100
XC098765.ogg,0.541200
```

> Requires coordinator to have been started with the correct keypair (`--db-path`). Returns `404` if no valid result exists yet.

---

## Validations

### `POST /validations`
Validator submits a validation report. Sets `verdict` on the result and marks job as `validated`.

**Body:**
```json
{
  "report_id": "...-val-1a2b",
  "job_id": "kaggle-acoustic_classification-b322",
  "result_id": "...",
  "validator_pubkey": "02ef...",
  "verdict": "valid",
  "recomputed_score": 0.8499,
  "score_delta": 0.0001,
  "notes": "score within tolerance (0.0001 < 0.0500)",
  "validator_sig": "3044...",
  "validated_at": 1773712600
}
```

---

## Payouts

### `GET /payouts`
List all payouts, optionally filtered by worker.

| Query param | Type   | Description            |
|-------------|--------|------------------------|
| `worker`    | string | Filter by worker pubkey |

```json
{
  "payouts": [
    {
      "payout_id": "6274a672-...",
      "job_id": "kaggle-acoustic_classification-b322",
      "worker_pubkey": "02abcd...",
      "amount_sompi": 85000000000,
      "txid": "0xbdae7c...",
      "paid_at": 1773712700
    }
  ]
}
```

> `amount_sompi = reward_sompi × recomputed_score` (floor 1 000 sompi for non-zero scores).

### `POST /payouts`
Settlement daemon registers a payout after anchoring on-chain. Marks job as `settled`.

---

## Datasets

### `GET /datasets/:job_id/files`
Lists audio files available for a job (served from coordinator's dataset cache).

```json
{
  "job_id": "kaggle-acoustic_classification-b322",
  "files": [
    { "filename": "train_audio/asbfly/XC134896.ogg", "size": 102400 },
    { "filename": "train_audio/comsan/XC201234.ogg", "size": 98304 }
  ],
  "total": 46207
}
```

### `GET /datasets/:job_id/download/*filename`
Downloads a single file from the dataset cache.

```
GET /datasets/kaggle-acoustic_classification-b322/download/train_audio/asbfly/XC134896.ogg
→ Content-Type: application/octet-stream
```

---

## Inference Scripts

### `GET /scripts/:task`
Returns the Python inference script for a task.

| Query param | Values                     | Default  | Description         |
|-------------|----------------------------|----------|---------------------|
| `backend`   | `yamnet` / `efficientnet`  | `yamnet` | Script variant      |

- `acoustic_classification?backend=yamnet` → `yamnet_infer.py`
- `acoustic_classification?backend=efficientnet` → `efficientnet_infer.py`

macOS miners automatically request `efficientnet`.

### `GET /scripts/:task/requirements`
Returns `requirements.txt` for the requested backend as `text/plain`.

```
GET /scripts/acoustic_classification/requirements?backend=yamnet
→ tensorflow>=2.13.0
   tensorflow-hub>=0.14.0
   librosa>=0.10.0
   numpy>=1.24.0
   soundfile>=0.12.0
```

---

## Job Status Flow

```
open → claimed → completed → validated → settled
         ↑            ↑           ↑          ↑
      POST /claim  POST /results  POST /validations  POST /payouts
```

---

## Reward Formula

```
amount_sompi = max(reward_sompi × recomputed_score, 1_000)
             = 0  if recomputed_score == 0
```

BirdCLEF default: `reward_sompi = 100_000_000_000` (100k XEN).
