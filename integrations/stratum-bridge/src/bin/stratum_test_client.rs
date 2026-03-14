//! stratum-test-client — manual Stratum v1 test against xenom-stratum-bridge
//!
//! Flow:
//!   1. TCP connect to pool
//!   2. mining.subscribe
//!   3. mining.authorize  (address.worker)
//!   4. Receive mining.notify — decode + print L2 job (param[6])
//!   5. Submit a dummy share (will be rejected, but tests the full round-trip)

use anyhow::{bail, Context, Result};
use clap::{Arg, Command};
use serde_json::{json, Value};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    time::{timeout, Duration},
};

// ── Helpers ────────────────────────────────────────────────────────────────────

fn fmt_hashrate(h: f64) -> String {
    let units = ["H/s", "KH/s", "MH/s", "GH/s", "TH/s"];
    let mut v = h;
    let mut i = 0;
    while v >= 1000.0 && i < units.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    format!("{:.2} {}", v, units[i])
}

async fn send(stream: &mut tokio::net::tcp::OwnedWriteHalf, msg: &Value) -> Result<()> {
    let mut line = serde_json::to_string(msg)?;
    line.push('\n');
    stream.write_all(line.as_bytes()).await?;
    println!("\x1b[90m  → {}\x1b[0m", line.trim_end());
    Ok(())
}

fn banner() {
    println!();
    println!("\x1b[32m╔══════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[32m║   Xenom Stratum Test Client  — genetics pool     ║\x1b[0m");
    println!("\x1b[32m╚══════════════════════════════════════════════════╝\x1b[0m");
    println!();
}

fn print_l2(job: &Value) {
    if job.is_null() {
        println!("  \x1b[33m[L2]\x1b[0m No L2 job dispatched (null)");
        return;
    }
    println!("  \x1b[36m╔══ L2 Genomics Job ══════════════════════════╗\x1b[0m");
    for key in &["theme", "job_id", "task", "dataset", "fragment", "reward_sompi"] {
        if let Some(v) = job.get(key) {
            let display = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            println!("  \x1b[36m║\x1b[0m  \x1b[90m{:<14}\x1b[0m \x1b[97m{}\x1b[0m", key, display);
        }
    }
    println!("  \x1b[36m╚═════════════════════════════════════════════╝\x1b[0m");
}

// ── Main ───────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let m = Command::new("stratum-test-client")
        .about("Stratum v1 test client for xenom-stratum-bridge")
        .arg(
            Arg::new("pool")
                .long("pool").short('p').value_name("HOST:PORT")
                .default_value("127.0.0.1:5555")
                .help("Stratum pool address"),
        )
        .arg(
            Arg::new("address")
                .long("address").short('a').value_name("XENOM_ADDR")
                .default_value("xenomdev:qzpztex736jkyax4q6mwhyfajkvp33hrsvy053hw46vmqn45gsdh6jgf0s7a0")
                .help("Mining reward address"),
        )
        .arg(
            Arg::new("worker")
                .long("worker").short('w').value_name("NAME")
                .default_value("test-worker-1")
                .help("Worker name"),
        )
        .arg(
            Arg::new("notify-count")
                .long("notify-count").short('n').value_name("N")
                .default_value("3")
                .value_parser(clap::value_parser!(usize))
                .help("Number of mining.notify messages to receive before exiting"),
        )
        .get_matches();

    let pool_addr    = m.get_one::<String>("pool").unwrap();
    let xenom_addr   = m.get_one::<String>("address").unwrap();
    let worker_name  = m.get_one::<String>("worker").unwrap();
    let notify_count = *m.get_one::<usize>("notify-count").unwrap();
    let username     = format!("{xenom_addr}.{worker_name}");

    banner();
    println!("  Pool    : \x1b[32m{pool_addr}\x1b[0m");
    println!("  Address : \x1b[32m{xenom_addr}\x1b[0m");
    println!("  Worker  : \x1b[32m{worker_name}\x1b[0m");
    println!();

    // ── TCP connect ────────────────────────────────────────────────────────────
    print!("  Connecting to {pool_addr} … ");
    let stream = TcpStream::connect(pool_addr)
        .await
        .context("Cannot connect to pool — is xenom-stratum-bridge running?")?;
    println!("\x1b[32mOK\x1b[0m");

    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half).lines();

    // ── 1. mining.subscribe ────────────────────────────────────────────────────
    println!("\n\x1b[33m[1] mining.subscribe\x1b[0m");
    send(
        &mut write_half,
        &json!({"id":1,"method":"mining.subscribe","params":["stratum-test-client/0.1"]}),
    )
    .await?;

    // read subscribe response
    let sub_resp = timeout(Duration::from_secs(10), reader.next_line())
        .await
        .context("Timeout waiting for subscribe response")?
        .context("Connection closed")?
        .context("Empty line")?;
    let sub: Value = serde_json::from_str(&sub_resp)?;
    println!("  \x1b[90m← {}\x1b[0m", sub_resp);

    // extract extranonce1
    let extranonce1 = sub["result"][1].as_str().unwrap_or("00000000");
    println!("  extranonce1 = \x1b[97m{extranonce1}\x1b[0m");

    // ── 2. mining.authorize ────────────────────────────────────────────────────
    println!("\n\x1b[33m[2] mining.authorize  ({username})\x1b[0m");
    send(
        &mut write_half,
        &json!({"id":2,"method":"mining.authorize","params":[username,"x"]}),
    )
    .await?;

    // drain responses until we see the authorize result
    loop {
        let line = timeout(Duration::from_secs(10), reader.next_line())
            .await
            .context("Timeout waiting for authorize")?
            .context("Connection closed")?
            .context("Empty")?;
        let v: Value = serde_json::from_str(&line)?;
        println!("  \x1b[90m← {}\x1b[0m", line);
        if v.get("id") == Some(&json!(2)) {
            if v["result"].as_bool() == Some(true) {
                println!("  \x1b[32mAuthorized ✓\x1b[0m");
            } else {
                bail!("Authorization failed: {}", v["error"]);
            }
            break;
        }
    }

    // ── 3. Receive mining.notify messages ──────────────────────────────────────
    println!("\n\x1b[33m[3] Waiting for {notify_count} mining.notify …\x1b[0m");
    let mut received = 0usize;
    let mut last_job_id = String::new();
    let mut last_bits = String::new();

    while received < notify_count {
        let line = timeout(Duration::from_secs(60), reader.next_line())
            .await
            .context("Timeout waiting for mining.notify")?
            .context("Connection closed")?
            .context("Empty")?;

        let v: Value = serde_json::from_str(&line)?;

        if v["method"].as_str() != Some("mining.notify") {
            println!("  \x1b[90m← (other) {}\x1b[0m", line);
            continue;
        }

        received += 1;
        let params = &v["params"];
        let job_id       = params[0].as_str().unwrap_or("?").to_owned();
        let pre_pow_hash = params[1].as_str().unwrap_or("?");
        let bits         = params[2].as_str().unwrap_or("?").to_owned();
        let epoch_seed   = params[3].as_str().unwrap_or("?");
        let timestamp_ms = params[4].as_u64().unwrap_or(0);
        let clean_jobs   = params[5].as_bool().unwrap_or(false);
        let l2_job       = &params[6];

        println!(
            "\n  \x1b[32m[notify #{received}]\x1b[0m  job_id=\x1b[97m{job_id}\x1b[0m  \
             bits=\x1b[97m{bits}\x1b[0m  clean=\x1b[97m{clean_jobs}\x1b[0m  ts=\x1b[97m{timestamp_ms}\x1b[0m"
        );
        println!("  pre_pow  : \x1b[90m{pre_pow_hash}\x1b[0m");
        println!("  epoch    : \x1b[90m{epoch_seed}\x1b[0m");
        print_l2(l2_job);

        last_job_id = job_id;
        last_bits   = bits;
    }

    // ── 4. Submit a dummy share (expect rejection — tests round-trip) ──────────
    println!("\n\x1b[33m[4] Submitting dummy share (expect rejection)\x1b[0m");
    let nonce_hex = "0000000000000000";
    let en2_hex   = "00000000";
    send(
        &mut write_half,
        &json!({
            "id": 4,
            "method": "mining.submit",
            "params": [username, last_job_id, en2_hex, nonce_hex]
        }),
    )
    .await?;

    // read submit response
    let submit_resp = timeout(Duration::from_secs(10), reader.next_line())
        .await
        .context("Timeout waiting for submit response")?
        .context("Connection closed")?
        .context("Empty")?;
    let sr: Value = serde_json::from_str(&submit_resp)?;
    println!("  \x1b[90m← {}\x1b[0m", submit_resp);
    if sr["result"].as_bool() == Some(true) {
        println!("  \x1b[32mShare accepted ✓\x1b[0m");
    } else {
        let err = &sr["error"];
        println!("  \x1b[33mShare rejected (expected for dummy nonce): {err}\x1b[0m");
    }

    // ── Summary ────────────────────────────────────────────────────────────────
    println!();
    println!("\x1b[32m══ Test complete ═══════════════════════════════════\x1b[0m");
    println!("  Pool   : {pool_addr}");
    println!("  Worker : {username}");
    println!("  Jobs   : {received} mining.notify received");
    println!("  Bits   : {last_bits}  (~{})", fmt_hashrate(
        u64::from_str_radix(last_bits.trim_start_matches("0x"), 16).unwrap_or(0) as f64
    ));
    println!("\x1b[32m════════════════════════════════════════════════════\x1b[0m");
    println!();

    Ok(())
}
