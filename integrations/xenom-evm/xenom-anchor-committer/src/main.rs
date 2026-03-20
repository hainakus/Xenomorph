use std::{path::Path, sync::Arc, time::{SystemTime, UNIX_EPOCH}};

use anyhow::{Context, Result};
use clap::{Arg, Command};
use kaspa_addresses::Prefix;
use kaspa_rpc_core::api::rpc::RpcApi;
use rocksdb::{ColumnFamilyDescriptor, Options, DB};
use serde::{Deserialize, Serialize};
use xenom_evm_core::L1CheckpointV1;

// ── Commitment payload layout (53 bytes) ─────────────────────────────────────
//
//  Offset  Len  Field
//  0       4    magic  = b"XEVM"
//  4       1    version = 0x01
//  5       32   checkpoint_id  (keccak256 of L1CheckpointV1 bytes)
//  37      8    block_number   (BE u64)
//  45      8    chain_id       (BE u64)
//                              = 53 bytes total

const COMMIT_MAGIC: &[u8; 4] = b"XEVM";
const COMMIT_VERSION: u8 = 0x01;
pub const COMMIT_PAYLOAD_SIZE: usize = 53;

fn build_commit_payload(
    checkpoint_id: &[u8; 32],
    block_number: u64,
    chain_id: u64,
) -> [u8; COMMIT_PAYLOAD_SIZE] {
    let mut out = [0u8; COMMIT_PAYLOAD_SIZE];
    out[0..4].copy_from_slice(COMMIT_MAGIC);
    out[4] = COMMIT_VERSION;
    out[5..37].copy_from_slice(checkpoint_id);
    out[37..45].copy_from_slice(&block_number.to_be_bytes());
    out[45..53].copy_from_slice(&chain_id.to_be_bytes());
    out
}

// ── CommitRecord ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CommitRecord {
    block_number: u64,
    checkpoint_id: String,
    l1_tx_ref: String,
    daa_score: u64,
    status: String,
    created_at_ms: u64,
}

// ── CommitterDb ───────────────────────────────────────────────────────────────

const CF_COMMIT_STATUS: &str = "commit_status";

struct CommitterDb {
    db: Arc<DB>,
}

impl CommitterDb {
    fn open(state_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(state_dir).context("create committer state dir")?;
        let db_path = state_dir.join("anchor-committer.rocksdb");

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf = ColumnFamilyDescriptor::new(CF_COMMIT_STATUS, Options::default());
        let db = DB::open_cf_descriptors(&opts, db_path, vec![cf])
            .context("open committer RocksDB")?;

        Ok(Self { db: Arc::new(db) })
    }

    fn get(&self, block_number: u64) -> Result<Option<CommitRecord>> {
        let cf = self.db.cf_handle(CF_COMMIT_STATUS)
            .ok_or_else(|| anyhow::anyhow!("missing CF {CF_COMMIT_STATUS}"))?;
        match self.db.get_cf(cf, block_number.to_be_bytes()).context("rocksdb get")? {
            Some(bytes) => {
                let rec: CommitRecord =
                    bincode::deserialize(&bytes).context("deserialize CommitRecord")?;
                Ok(Some(rec))
            }
            None => Ok(None),
        }
    }

    fn put(&self, rec: &CommitRecord) -> Result<()> {
        let cf = self.db.cf_handle(CF_COMMIT_STATUS)
            .ok_or_else(|| anyhow::anyhow!("missing CF {CF_COMMIT_STATUS}"))?;
        let bytes = bincode::serialize(rec).context("serialize CommitRecord")?;
        self.db
            .put_cf(cf, rec.block_number.to_be_bytes(), bytes)
            .context("rocksdb put")
    }
}

// ── EVM checkpoint polling ────────────────────────────────────────────────────

async fn fetch_latest_checkpoint(
    http: &reqwest::Client,
    evm_url: &str,
) -> Result<Option<(L1CheckpointV1, String)>> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method":  "xenom_latestCheckpoint",
        "params":  [],
        "id":      1
    });

    let resp: serde_json::Value = http
        .post(evm_url)
        .json(&body)
        .send()
        .await
        .context("xenom_latestCheckpoint POST")?
        .json()
        .await
        .context("parse xenom_latestCheckpoint response")?;

    if let Some(err) = resp.get("error") {
        anyhow::bail!("xenom_latestCheckpoint RPC error: {err}");
    }

    let result = &resp["result"];
    if result.is_null() {
        return Ok(None);
    }

    let bytes_hex = result["bytes"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'bytes' in checkpoint response"))?
        .trim_start_matches("0x");

    let bytes = hex::decode(bytes_hex).context("decode checkpoint bytes hex")?;

    let checkpoint_id = result["checkpointId"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing 'checkpointId' in checkpoint response"))?
        .to_owned();

    let cp = L1CheckpointV1::from_bytes(&bytes)
        .ok_or_else(|| anyhow::anyhow!("invalid checkpoint bytes (got {} bytes)", bytes.len()))?;

    Ok(Some((cp, checkpoint_id)))
}

// ── Commit cycle ──────────────────────────────────────────────────────────────

async fn commit_cycle(
    http:       &reqwest::Client,
    evm_url:    &str,
    rpc:        Option<&Arc<kaspa_grpc_client::GrpcClient>>,
    keypair:    Option<&secp256k1::Keypair>,
    fee_sompi:  u64,
    prefix:     Prefix,
    db:         &CommitterDb,
    dry_run:    bool,
) -> Result<bool> {
    let Some((checkpoint, checkpoint_id)) = fetch_latest_checkpoint(http, evm_url).await? else {
        log::debug!("no checkpoint available yet on EVM node");
        return Ok(false);
    };

    let block_number = checkpoint.block_number;

    if let Some(rec) = db.get(block_number)? {
        log::debug!(
            "block {block_number}: already committed (status={} l1_tx={})",
            rec.status,
            rec.l1_tx_ref.get(..12.min(rec.l1_tx_ref.len())).unwrap_or(&rec.l1_tx_ref),
        );
        return Ok(false);
    }

    let id_bytes_vec =
        hex::decode(checkpoint_id.trim_start_matches("0x")).context("decode checkpoint_id")?;
    let mut id_arr = [0u8; 32];
    let copy_len = id_bytes_vec.len().min(32);
    id_arr[..copy_len].copy_from_slice(&id_bytes_vec[..copy_len]);

    let commit_payload =
        build_commit_payload(&id_arr, block_number, checkpoint.chain_id);

    log::info!(
        "Committing checkpoint block={block_number} id={}… ({COMMIT_PAYLOAD_SIZE}B payload)",
        &checkpoint_id[..14.min(checkpoint_id.len())],
    );

    if dry_run || rpc.is_none() || keypair.is_none() {
        log::info!("  dry-run: skipping L1 submission");
        let rec = CommitRecord {
            block_number,
            checkpoint_id,
            l1_tx_ref: "dry-run".to_owned(),
            daa_score: 0,
            status: "dry-run".to_owned(),
            created_at_ms: now_ms(),
        };
        db.put(&rec)?;
        return Ok(true);
    }

    let rpc_client = rpc.unwrap();
    let kp = keypair.unwrap();

    let daa_score = rpc_client
        .get_block_dag_info()
        .await
        .context("get_block_dag_info")?
        .virtual_daa_score;

    let l1_tx_ref =
        xenom_anchor_client::submit_anchor(rpc_client, kp, &commit_payload, fee_sompi)
            .await
            .context("submit_anchor to L1")?;

    log::info!("  l1_tx={l1_tx_ref}  daa_score={daa_score}");

    let rec = CommitRecord {
        block_number,
        checkpoint_id,
        l1_tx_ref,
        daa_score,
        status: "submitted".to_owned(),
        created_at_ms: now_ms(),
    };
    db.put(&rec)?;

    Ok(true)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn load_privkey_opt(env_var: &str, key_file: Option<&str>) -> Option<String> {
    if let Ok(val) = std::env::var(env_var) {
        let val = val.trim().to_string();
        if !val.is_empty() {
            return Some(val);
        }
    }
    if let Some(path) = key_file {
        if let Ok(val) = std::fs::read_to_string(path) {
            let val = val.trim().to_string();
            if !val.is_empty() {
                return Some(val);
            }
        }
    }
    None
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();

    let evm_raw = m
        .get_one::<String>("evm-node")
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:8545".to_owned());
    let evm_url = if evm_raw.starts_with("http") {
        evm_raw
    } else {
        format!("http://{evm_raw}")
    };

    let node_raw = m
        .get_one::<String>("node")
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:36669".to_owned());
    let node_url = if node_raw.starts_with("grpc://") {
        node_raw.clone()
    } else {
        format!("grpc://{node_raw}")
    };

    let state_dir = m
        .get_one::<String>("state-dir")
        .cloned()
        .unwrap_or_else(|| "/var/lib/xenom-evm-committer".to_owned());

    let poll_ms: u64 = m
        .get_one::<String>("poll-ms")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let dry_run = !m.get_flag("submit");

    let network_prefix = if m.get_flag("devnet") {
        Prefix::Devnet
    } else if m.get_flag("testnet") {
        Prefix::Testnet
    } else {
        Prefix::Mainnet
    };

    let fee_sompi: u64 = m
        .get_one::<String>("fee-sompi")
        .and_then(|s| s.parse().ok())
        .unwrap_or(xenom_anchor_client::DEFAULT_FEE_PER_INPUT);

    let privkey_hex = load_privkey_opt(
        "COMMITTER_PRIVKEY",
        m.get_one::<String>("key-file").map(|s| s.as_str()),
    );

    if !dry_run && privkey_hex.is_none() {
        anyhow::bail!(
            "--submit requires a private key: set $COMMITTER_PRIVKEY or use --key-file <PATH>"
        );
    }

    let keypair: Option<secp256k1::Keypair> = privkey_hex
        .as_deref()
        .map(xenom_anchor_client::keypair_from_hex)
        .transpose()
        .context("invalid private key (expected 64 hex chars)")?;

    log::info!("Xenom Anchor Committer starting");
    log::info!("  evm-node:  {evm_url}");
    log::info!("  l1-node:   {node_url}");
    log::info!("  state-dir: {state_dir}");
    log::info!("  network:   {network_prefix:?}");
    log::info!("  poll:      {poll_ms}ms");
    log::info!("  dry-run:   {dry_run}");
    if let Some(ref kp) = keypair {
        log::info!(
            "  funding:   {}",
            xenom_anchor_client::address_from_keypair(kp)
        );
    }

    let db = CommitterDb::open(Path::new(&state_dir)).context("open committer DB")?;
    let http = reqwest::Client::new();

    let rpc: Option<Arc<kaspa_grpc_client::GrpcClient>> = if !dry_run {
        let client = kaspa_grpc_client::GrpcClient::connect(node_url.clone())
            .await
            .context("connect to L1 node")?;
        log::info!("  connected to L1 node: {node_url}");
        Some(Arc::new(client))
    } else {
        log::info!("  dry-run: not connecting to L1 node");
        None
    };

    loop {
        match commit_cycle(
            &http,
            &evm_url,
            rpc.as_ref(),
            keypair.as_ref(),
            fee_sompi,
            network_prefix,
            &db,
            dry_run,
        )
        .await
        {
            Ok(true) => {}
            Ok(false) => {}
            Err(e) => log::warn!("commit cycle error: {e:#}"),
        }

        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_millis(poll_ms)) => {}
            _ = tokio::signal::ctrl_c() => {
                log::info!("Shutting down...");
                break;
            }
        }
    }

    Ok(())
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("xenom-anchor-committer")
        .about("Xenom EVM L2 — anchor committer: reads L2 checkpoints and publishes commitments to L1")
        .arg(
            Arg::new("evm-node")
                .long("evm-node")
                .value_name("URL")
                .default_value("127.0.0.1:8545")
                .help("Xenom EVM L2 JSON-RPC endpoint"),
        )
        .arg(
            Arg::new("node")
                .short('n')
                .long("node")
                .value_name("HOST:PORT")
                .default_value("127.0.0.1:36669")
                .help("Xenom L1 gRPC endpoint"),
        )
        .arg(
            Arg::new("state-dir")
                .long("state-dir")
                .value_name("PATH")
                .default_value("/var/lib/xenom-evm-committer")
                .help("Directory for committer state DB (anchor-committer.rocksdb)"),
        )
        .arg(
            Arg::new("poll-ms")
                .long("poll-ms")
                .value_name("MS")
                .default_value("10000")
                .help("Poll interval in milliseconds"),
        )
        .arg(
            Arg::new("submit")
                .long("submit")
                .action(clap::ArgAction::SetTrue)
                .help("Actually submit commitments to L1 (default: dry-run)"),
        )
        .arg(
            Arg::new("key-file")
                .short('k')
                .long("key-file")
                .value_name("PATH")
                .help(
                    "Path to file containing the secp256k1 private key (64 hex chars). \
                     Alternatively set $COMMITTER_PRIVKEY. Required with --submit.",
                ),
        )
        .arg(
            Arg::new("fee-sompi")
                .long("fee-sompi")
                .value_name("N")
                .default_value("2000")
                .help("Relay fee per input in sompi"),
        )
        .arg(
            Arg::new("devnet")
                .long("devnet")
                .action(clap::ArgAction::SetTrue)
                .help("Use devnet address prefix (xenomdev:)"),
        )
        .arg(
            Arg::new("testnet")
                .long("testnet")
                .action(clap::ArgAction::SetTrue)
                .help("Use testnet address prefix (xenomtest:)"),
        )
}
