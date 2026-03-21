use anyhow::{Context, Result};
use bioproof_core::AnchorPayload;
use clap::{Arg, Command};
use kaspa_grpc_client::GrpcClient;
use kaspa_rpc_core::api::rpc::RpcApi;
use sqlx::SqlitePool;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

// ── DB schema ─────────────────────────────────────────────────────────────────

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS anchors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    txid          TEXT    NOT NULL UNIQUE,
    block_hash    TEXT,
    daa_score     INTEGER,
    block_time    INTEGER,
    proof_root    TEXT    NOT NULL,
    manifest_hash TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    indexed_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_anchors_proof_root    ON anchors(proof_root);
CREATE INDEX IF NOT EXISTS idx_anchors_manifest_hash ON anchors(manifest_hash);
CREATE INDEX IF NOT EXISTS idx_anchors_kind          ON anchors(kind);
CREATE INDEX IF NOT EXISTS idx_anchors_daa_score     ON anchors(daa_score);

CREATE TABLE IF NOT EXISTS indexer_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

const STATE_KEY_LOW_HASH: &str = "low_hash";

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();
    let node_addr  = m.get_one::<String>("node").unwrap();
    let db_path    = m.get_one::<String>("db-path").unwrap();
    let poll_ms: u64 = m.get_one::<String>("poll-ms")
        .and_then(|s| s.parse().ok()).unwrap_or(2_000);

    // ── Database ─────────────────────────────────────────────────────────────
    let pool = SqlitePool::connect(&format!("sqlite:{db_path}?mode=rwc"))
        .await
        .with_context(|| format!("cannot open SQLite database '{db_path}'"))?;
    sqlx::raw_sql(SCHEMA).execute(&pool).await.context("schema init")?;
    log::info!("Database: {db_path}");

    // ── gRPC connection ───────────────────────────────────────────────────────
    let url = if node_addr.starts_with("grpc://") {
        node_addr.clone()
    } else {
        format!("grpc://{node_addr}")
    };
    let rpc = Arc::new(
        GrpcClient::connect(url)
            .await
            .context("cannot connect to Xenom node")?,
    );
    log::info!("Connected to {node_addr}");

    // ── Polling loop ──────────────────────────────────────────────────────────
    log::info!("Indexer running (poll every {poll_ms} ms)…");
    loop {
        if let Err(e) = poll_once(&rpc, &pool).await {
            log::warn!("poll error: {e:#}");
        }
        sleep(Duration::from_millis(poll_ms)).await;
    }
}

// ── Polling logic ─────────────────────────────────────────────────────────────

async fn poll_once(rpc: &Arc<GrpcClient>, pool: &SqlitePool) -> Result<()> {
    // Fetch the low hash cursor (start of un-indexed range).
    let low_hash_str: Option<String> = sqlx::query_scalar(
        "SELECT value FROM indexer_state WHERE key = ?1",
    )
    .bind(STATE_KEY_LOW_HASH)
    .fetch_optional(pool)
    .await?;

    let low_hash = low_hash_str
        .as_deref()
        .and_then(|s| s.parse().ok());

    let resp = rpc.get_blocks(low_hash, true, true).await?;
    if resp.blocks.is_empty() {
        return Ok(());
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    for block in &resp.blocks {
        let block_hash = format!("{}", block.header.hash);
        let daa_score  = block.header.daa_score as i64;
        let block_time = (block.header.timestamp / 1000) as i64; // ms → s

        for tx in &block.transactions {
            // BioProof anchors are embedded in the native Kaspa transaction
            // payload field (not OP_RETURN — simpler and more idiomatic).
            if let Some(payload) = extract_bioproof_payload(&tx.payload) {
                let txid = tx
                    .verbose_data
                    .as_ref()
                    .map(|v| format!("{}", v.transaction_id))
                    .unwrap_or_default();

                if txid.is_empty() {
                    continue;
                }

                log::info!("Anchor found: txid={txid}  proof_root={}", payload.proof_root);

                sqlx::query(
                    "INSERT OR IGNORE INTO anchors
                     (txid, block_hash, daa_score, block_time, proof_root, manifest_hash, kind, indexed_at)
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                )
                .bind(&txid)
                .bind(&block_hash)
                .bind(daa_score)
                .bind(block_time)
                .bind(&payload.proof_root)
                .bind(&payload.manifest_hash)
                .bind(&payload.kind)
                .bind(now)
                .execute(pool)
                .await?;
            }
        }

        // Advance cursor past this block.
        sqlx::query(
            "INSERT OR REPLACE INTO indexer_state (key, value) VALUES (?1, ?2)",
        )
        .bind(STATE_KEY_LOW_HASH)
        .bind(&block_hash)
        .execute(pool)
        .await?;
    }

    log::debug!("Scanned {} block(s)", resp.blocks.len());
    Ok(())
}

// ── BioProof payload extraction ───────────────────────────────────────────────

/// Parse a transaction payload and decode a BioProof anchor if present.
/// BioProof anchors are stored in the native Kaspa `tx.payload` as compact JSON.
fn extract_bioproof_payload(payload: &[u8]) -> Option<AnchorPayload> {
    AnchorPayload::from_op_return_bytes(payload)
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("bioproof-indexer")
        .about("BioProof indexer — scans the Xenom chain and indexes bioproof anchor transactions")
        .arg(Arg::new("node")
            .short('n').long("node").value_name("ADDR")
            .default_value("grpc://localhost:36669")
            .help("Xenom node gRPC address"))
        .arg(Arg::new("db-path")
            .short('d').long("db-path").value_name("PATH")
            .default_value("bioproof.db")
            .help("SQLite database file"))
        .arg(Arg::new("poll-ms")
            .long("poll-ms").value_name("MS")
            .default_value("2000")
            .help("Chain polling interval in milliseconds"))
}
