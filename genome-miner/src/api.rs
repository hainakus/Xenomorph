// ── HiveOS-compatible stats API ───────────────────────────────────────────────
//
// Runs a minimal HTTP server (default port 4000) that returns a JSON payload
// matching the HiveOS miner stats format.  HiveOS polls this endpoint every
// few seconds via the `stats_url` field in the miner flight-sheet config.
//
// GPU hardware metrics (temp / fan / power) are read from:
//   1. nvidia-smi  (NVIDIA GPUs, all platforms)
//   2. rocm-smi    (AMD GPUs via ROCm)
//   3. Linux sysfs hwmon  (AMD GPUs without ROCm, or any hwmon-exposed GPU)
//
// A background task (`hw_poll_task`) refreshes hardware metrics every 5 s
// and writes them back into `DashStats::gpus`, so the TUI also shows them.
//
// HiveOS JSON format reference:
//   https://github.com/nicehash/NiceHashQuickMiner/blob/master/NiceHashQuickMiner/
//   (and community-documented at https://hiveon.com/knowledge-base/hive_miner_api/)
//
// Field semantics expected by HiveOS:
//   hs        – hashrate array, one entry per GPU, in H/s
//   hs_units  – "hs" | "khs" | "mhs" | "ghs"
//   temp      – GPU temperature array in °C
//   fan       – GPU fan-speed array in %
//   power     – GPU power-draw array in W
//   accepted  – total accepted shares
//   rejected  – total rejected shares
//   algo      – algorithm name string
//   ver       – miner version string
//   uptime    – seconds since miner start
//   ar        – [accepted, rejected] (duplicate for some HiveOS consumers)

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde::Serialize;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    time::sleep,
};

use crate::tui::DashStats;

// ── HiveOS JSON payload ───────────────────────────────────────────────────────

#[derive(Serialize)]
struct HiveStats<'a> {
    hs:       Vec<f64>,
    hs_units: &'static str,
    temp:     Vec<u32>,
    fan:      Vec<u32>,
    power:    Vec<f64>,
    accepted: u64,
    rejected: u64,
    algo:     &'a str,
    ver:      &'static str,
    uptime:   u64,
    ar:       [u64; 2],
}

// ── Per-GPU hardware metrics ──────────────────────────────────────────────────

#[derive(Clone, Default, Debug)]
pub struct GpuHwStats {
    pub temp:  u32,   // °C
    pub fan:   u32,   // %
    pub power: f64,   // W
}

// ── NVIDIA via nvidia-smi ─────────────────────────────────────────────────────

/// Queries all NVIDIA GPUs via `nvidia-smi`.
/// Returns `None` if nvidia-smi is not installed or reports no GPUs.
fn query_nvidia_smi(count: usize) -> Option<Vec<GpuHwStats>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,temperature.gpu,fan.speed,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }

    let text = std::str::from_utf8(&output.stdout).ok()?;
    let mut entries: Vec<(usize, GpuHwStats)> = Vec::new();

    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 4 {
            continue;
        }
        let idx: usize = parts[0].parse().unwrap_or(0);
        let temp:  u32  = parts[1].parse().unwrap_or(0);
        // fan.speed may be "[N/A]" on headless/server GPUs — treat as 0
        let fan:   u32  = parts[2].parse().unwrap_or(0);
        let power: f64  = parts[3].parse().unwrap_or(0.0);
        entries.push((idx, GpuHwStats { temp, fan, power }));
    }

    if entries.is_empty() {
        return None;
    }

    entries.sort_by_key(|(i, _)| *i);
    let mut out: Vec<GpuHwStats> = (0..count).map(|_| GpuHwStats::default()).collect();
    for (idx, hw) in entries {
        if idx < count {
            out[idx] = hw;
        }
    }
    Some(out)
}

// ── AMD via rocm-smi ──────────────────────────────────────────────────────────

/// Queries AMD GPUs via `rocm-smi --showtemp --showfan --showpower --csv`.
/// CSV columns: device, Temperature (Sensor junction) (C), Fan speed (%), Power (W)
fn query_rocm_smi(count: usize) -> Option<Vec<GpuHwStats>> {
    let output = std::process::Command::new("rocm-smi")
        .args(["--showtemp", "--showfan", "--showpower", "--csv"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = std::str::from_utf8(&output.stdout).ok()?;
    let mut out: Vec<GpuHwStats> = (0..count).map(|_| GpuHwStats::default()).collect();
    let mut device_idx = 0usize;

    for line in text.lines().skip(1) { // skip CSV header
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 4 {
            continue;
        }
        if device_idx < count {
            out[device_idx] = GpuHwStats {
                temp:  parts[1].parse().unwrap_or(0),
                fan:   parts[2].parse().unwrap_or(0),
                power: parts[3].parse().unwrap_or(0.0),
            };
        }
        device_idx += 1;
    }

    if device_idx == 0 {
        None
    } else {
        Some(out)
    }
}

// ── AMD / generic via Linux sysfs hwmon ──────────────────────────────────────

/// Reads GPU hardware metrics from `/sys/class/hwmon/hwmon*/` on Linux.
/// Works for `amdgpu` kernel driver (and any driver that exposes standard
/// hwmon attributes).  Falls back gracefully if files are absent.
#[cfg(target_os = "linux")]
fn query_sysfs_hwmon(count: usize) -> Option<Vec<GpuHwStats>> {
    use std::fs;

    let hwmon_base = std::path::Path::new("/sys/class/hwmon");
    let mut hw_dirs: Vec<std::path::PathBuf> = fs::read_dir(hwmon_base)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let p    = e.path();
            let name = fs::read_to_string(p.join("name")).unwrap_or_default();
            // Accept both amdgpu and generic "gpu" named hwmon entries
            let n = name.trim();
            if n == "amdgpu" || n.contains("gpu") {
                Some(p)
            } else {
                None
            }
        })
        .collect();

    hw_dirs.sort();
    if hw_dirs.is_empty() {
        return None;
    }

    let mut out: Vec<GpuHwStats> = (0..count).map(|_| GpuHwStats::default()).collect();

    for (i, hw_path) in hw_dirs.iter().enumerate() {
        if i >= count {
            break;
        }

        // Temperature: temp1_input is in millidegrees Celsius
        let temp_mc: u64 = fs::read_to_string(hw_path.join("temp1_input"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        // Fan: fan1_input (RPM) + fan1_max (max RPM) → percentage
        let fan_rpm: u64 = fs::read_to_string(hw_path.join("fan1_input"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        let fan_max: u64 = fs::read_to_string(hw_path.join("fan1_max"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        // Power: power1_average is in microwatts
        let power_uw: u64 = fs::read_to_string(hw_path.join("power1_average"))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        out[i] = GpuHwStats {
            temp:  (temp_mc / 1_000) as u32,
            fan:   if fan_max > 0 { ((fan_rpm * 100) / fan_max) as u32 } else { 0 },
            power: power_uw as f64 / 1_000_000.0,
        };
    }

    Some(out)
}

#[cfg(not(target_os = "linux"))]
fn query_sysfs_hwmon(_count: usize) -> Option<Vec<GpuHwStats>> {
    None
}

// ── Unified hardware query ────────────────────────────────────────────────────

/// Returns hardware stats for `count` GPUs.
/// Tries NVIDIA → ROCm → sysfs in order; fills zeros for unknown values.
pub fn query_hw_stats(count: usize) -> Vec<GpuHwStats> {
    if count == 0 {
        return Vec::new();
    }
    query_nvidia_smi(count)
        .or_else(|| query_rocm_smi(count))
        .or_else(|| query_sysfs_hwmon(count))
        .unwrap_or_else(|| (0..count).map(|_| GpuHwStats::default()).collect())
}

// ── Background hardware-polling task ─────────────────────────────────────────

/// Polls GPU hardware metrics every 5 seconds and writes them into `DashStats`.
/// Spawn this as a background tokio task alongside the mining tasks.
pub async fn hw_poll_task(stats: Arc<Mutex<DashStats>>) {
    loop {
        sleep(Duration::from_secs(5)).await;

        let count = {
            stats.lock().unwrap().gpus.len()
        };
        if count == 0 {
            continue;
        }

        let hw = query_hw_stats(count);

        let mut s = stats.lock().unwrap();
        for (i, h) in hw.iter().enumerate() {
            if let Some(g) = s.gpus.get_mut(i) {
                g.temp  = h.temp;
                g.fan   = h.fan;
                g.power = h.power;
            }
        }
    }
}

// ── Minimal HTTP server ───────────────────────────────────────────────────────

/// Starts a simple HTTP/1.1 server on `0.0.0.0:<port>` that serves a single
/// HiveOS-compatible JSON stats payload on any GET request.
///
/// Configure in your HiveOS flight sheet as:
///   API endpoint: http://<rig-ip>:<port>/
pub async fn run_api_server(
    port:  u16,
    stats: Arc<Mutex<DashStats>>,
    start: Instant,
) {
    let addr = format!("0.0.0.0:{port}");
    let listener = match TcpListener::bind(&addr).await {
        Ok(l)  => l,
        Err(e) => {
            eprintln!("[api] Failed to bind {addr}: {e}");
            return;
        }
    };

    // Print directly so the operator sees the port even when TUI silences logs.
    eprintln!("[api] HiveOS stats API on http://0.0.0.0:{port}/");

    loop {
        let (mut stream, _peer) = match listener.accept().await {
            Ok(c)  => c,
            Err(_) => continue,
        };

        let stats2 = stats.clone();
        let start2  = start;

        tokio::spawn(async move {
            // Consume the HTTP request (we don't care about the path/method).
            let mut req_buf = [0u8; 512];
            let _ = stream.read(&mut req_buf).await;

            let json = build_json(&stats2, start2);

            let response = format!(
                concat!(
                    "HTTP/1.1 200 OK\r\n",
                    "Content-Type: application/json\r\n",
                    "Content-Length: {}\r\n",
                    "Connection: close\r\n",
                    "\r\n",
                    "{}",
                ),
                json.len(),
                json,
            );

            let _ = stream.write_all(response.as_bytes()).await;
        });
    }
}

/// Serialises the current `DashStats` snapshot into HiveOS-compatible JSON.
fn build_json(stats: &Arc<Mutex<DashStats>>, start: Instant) -> String {
    let s = stats.lock().unwrap();

    // H/s per GPU (miner stores MH/s internally)
    let hs: Vec<f64>  = s.gpus.iter().map(|g| g.hashrate * 1_000_000.0).collect();
    let temp: Vec<u32> = s.gpus.iter().map(|g| g.temp).collect();
    let fan:  Vec<u32> = s.gpus.iter().map(|g| g.fan).collect();
    let power: Vec<f64> = s.gpus.iter().map(|g| g.power).collect();

    let algo = "genome-pow";

    let payload = HiveStats {
        hs,
        hs_units: "hs",
        temp,
        fan,
        power,
        accepted: s.accepted,
        rejected: s.rejected,
        algo,
        ver:    env!("CARGO_PKG_VERSION"),
        uptime: start.elapsed().as_secs(),
        ar:     [s.accepted, s.rejected],
    };

    serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_owned())
}
