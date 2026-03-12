use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Incoming JSON-RPC message from a miner.
#[derive(Debug, Deserialize)]
pub struct StratumRequest {
    pub id: Option<Value>,
    pub method: String,
    pub params: Value,
}

/// Outgoing JSON-RPC response to a miner.
#[derive(Debug, Serialize)]
pub struct StratumResponse {
    pub id: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<Value>,
}

/// Outgoing JSON-RPC notification to a miner (id = null).
#[derive(Debug, Serialize)]
pub struct StratumNotification {
    pub id: Option<Value>,
    pub method: String,
    pub params: Value,
}

impl StratumResponse {
    pub fn ok(id: Option<Value>, result: Value) -> Self {
        Self { id, result: Some(result), error: None }
    }

    pub fn err(id: Option<Value>, code: i32, msg: &str) -> Self {
        Self { id, result: None, error: Some(serde_json::json!([code, msg, null])) }
    }
}

impl StratumNotification {
    /// `mining.set_difficulty` – informs the miner of the current share difficulty.
    pub fn set_difficulty(difficulty: f64) -> Self {
        Self { id: None, method: "mining.set_difficulty".into(), params: serde_json::json!([difficulty]) }
    }

    /// `mining.notify` – sends new work to the miner.
    ///
    /// Params (Xenom Genome PoW extension):
    /// 1. job_id          — hex string
    /// 2. pre_pow_hash    — 64-char hex (32 bytes): `hash_override_nonce_time(header, 0, 0)`
    /// 3. bits            — 8-char hex (4 bytes): compact difficulty target
    /// 4. epoch_seed      — 64-char hex (32 bytes): genome epoch seed
    /// 5. timestamp       — 16-char hex (8 bytes): template timestamp in milliseconds
    /// 6. clean_jobs      — bool: true → abandon previous jobs
    pub fn notify(
        job_id: &str,
        pre_pow_hash: &str,
        bits: &str,
        epoch_seed: &str,
        timestamp: &str,
        clean_jobs: bool,
    ) -> Self {
        Self {
            id: None,
            method: "mining.notify".into(),
            params: serde_json::json!([job_id, pre_pow_hash, bits, epoch_seed, timestamp, clean_jobs]),
        }
    }
}
