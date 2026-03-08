use std::{
    collections::VecDeque,
    io,
    process::Command,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Terminal,
};

const LOG_CAP: usize = 200;

pub struct GpuStats {
    pub id: usize,
    pub name: String,
    pub hashrate: f64,
    pub accepted: u64,
    pub rejected: u64,

    pub temp: u32,
    pub fan: u32,
    pub util: u32,
    pub power: f32,

    pub mem_used: u32,
    pub mem_total: u32,
}

pub struct DashStats {
    pub rpcserver: String,
    pub connected: bool,
    pub mode: String,
    pub daa_score: u64,
    pub bits: u32,
    pub genome_active: bool,

    pub total_mhs: f64,
    pub accepted: u64,
    pub rejected: u64,

    pub gpus: Vec<GpuStats>,

    pub log: VecDeque<String>,
    start: Instant,
}

impl DashStats {

    pub fn new(rpcserver: String, mode: String, _threads: usize) -> Self {

        Self {
            rpcserver,
            connected: false,
            mode,
            daa_score: 0,
            bits: 0,
            genome_active: false,

            total_mhs: 0.0,
            accepted: 0,
            rejected: 0,

            gpus: Vec::new(),

            log: VecDeque::with_capacity(LOG_CAP),
            start: Instant::now(),
        }
    }

    pub fn uptime(&self) -> String {

        let s = self.start.elapsed().as_secs();

        format!(
            "{:02}:{:02}:{:02}",
            s / 3600,
            (s / 60) % 60,
            s % 60
        )
    }

    pub fn push_log(&mut self, msg: String) {

        if self.log.len() >= LOG_CAP {
            self.log.pop_front();
        }

        self.log.push_back(msg);
    }
}

fn fmt_hashrate(mhs: f64) -> String {

    if mhs >= 1000.0 {
        format!("{:.2} GH/s", mhs / 1000.0)
    } else {
        format!("{:.2} MH/s", mhs)
    }
}

fn bar(v: f64, max: f64, width: usize) -> String {

    let filled = ((v / max) * width as f64).round() as usize;

    format!(
        "[{}{}]",
        "█".repeat(filled.min(width)),
        "░".repeat(width - filled.min(width))
    )
}

fn temp_color(t: u32) -> Color {

    match t {
        0..=59 => Color::Green,
        60..=69 => Color::Yellow,
        70..=79 => Color::LightRed,
        _ => Color::Red,
    }
}

fn util_color(u: u32) -> Color {

    match u {
        0..=40 => Color::Gray,
        41..=70 => Color::Cyan,
        71..=90 => Color::Green,
        _ => Color::Red,
    }
}

fn power_color(p: f32) -> Color {

    match p {
        0.0..=200.0 => Color::Green,
        200.0..=300.0 => Color::Yellow,
        _ => Color::Red,
    }
}

/* GPU monitoring — multi-vendor */

fn is_nvidia(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("nvidia") || n.contains("geforce") || n.contains("quadro") || n.contains("tesla")
}

fn is_amd(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("amd") || n.contains("radeon") || n.contains("gfx") || n.contains("vega") || n.contains("navi")
}

fn is_intel(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("intel") || n.contains("arc") || n.contains("iris") || n.contains("uhd graphics")
}

fn names_match(a: &str, b: &str) -> bool {
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();
    a.contains(b.as_str()) || b.contains(a.as_str())
}

fn update_gpu(stats: &mut DashStats) {
    update_nvidia_smi(stats);
    update_rocm_smi(stats);
    update_intel(stats);
}

fn update_nvidia_smi(stats: &mut DashStats) {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,temperature.gpu,fan.speed,power.draw,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(out) = out else { return };
    let text = String::from_utf8_lossy(&out.stdout);

    for line in text.lines() {
        let p: Vec<&str> = line.split(", ").collect();
        if p.len() < 8 { continue; }
        let smi_name = p[1].trim();
        if let Some(g) = stats.gpus.iter_mut()
            .find(|g| is_nvidia(&g.name) && names_match(&g.name, smi_name))
        {
            g.temp      = p[2].trim().parse().unwrap_or(g.temp);
            g.fan       = p[3].trim().parse().unwrap_or(g.fan);
            g.power     = p[4].trim().parse().unwrap_or(g.power);
            g.util      = p[5].trim().parse().unwrap_or(g.util);
            g.mem_used  = p[6].trim().parse().unwrap_or(g.mem_used);
            g.mem_total = p[7].trim().parse().unwrap_or(g.mem_total);
        }
    }
}

fn update_rocm_smi(stats: &mut DashStats) {
    let out = Command::new("rocm-smi")
        .args([
            "--showproductname",
            "--showtemp",
            "--showfan",
            "--showuse",
            "--showpower",
            "--showmeminfo", "vram",
            "--json",
        ])
        .output();

    let Ok(out) = out else { return };
    let text = String::from_utf8_lossy(&out.stdout);
    let cards = parse_rocm_json(&text);

    let amd_idxs: Vec<usize> = stats.gpus.iter()
        .enumerate()
        .filter(|(_, g)| is_amd(&g.name))
        .map(|(i, _)| i)
        .collect();

    for (card_num, kv) in cards {
        let stats_idx = kv.iter()
            .find(|(k, _)| k.contains("card series") || k.contains("product name"))
            .and_then(|(_, v)| stats.gpus.iter().position(|g| is_amd(&g.name) && names_match(&g.name, v)))
            .or_else(|| amd_idxs.get(card_num).copied());

        let Some(idx) = stats_idx else { continue };
        let g = &mut stats.gpus[idx];

        for (k, v) in &kv {
            let k = k.as_str();
            if k.contains("temperature") {
                g.temp  = v.parse().unwrap_or(g.temp);
            } else if k.contains("fan speed (%)") {
                g.fan   = v.parse().unwrap_or(g.fan);
            } else if k.contains("gpu use") {
                g.util  = v.parse().unwrap_or(g.util);
            } else if k.contains("power") && !k.contains("cap") {
                g.power = v.parse().unwrap_or(g.power);
            } else if k.contains("vram total memory") && !k.contains("used") {
                if let Ok(b) = v.parse::<u64>() { g.mem_total = (b / 1_048_576) as u32; }
            } else if k.contains("vram total used") {
                if let Ok(b) = v.parse::<u64>() { g.mem_used = (b / 1_048_576) as u32; }
            }
        }
    }
}

fn update_intel(stats: &mut DashStats) {
    let intel_idxs: Vec<usize> = stats.gpus.iter()
        .enumerate()
        .filter(|(_, g)| is_intel(&g.name))
        .map(|(i, _)| i)
        .collect();
    if intel_idxs.is_empty() { return; }

    // Try xpu-smi (Intel Data Center / Arc)
    let out = Command::new("xpu-smi")
        .args(["dump", "-m", "0,1,2,5,18,19"])
        .output();

    if let Ok(out) = out {
        let text = String::from_utf8_lossy(&out.stdout);
        let mut card = 0usize;
        for line in text.lines() {
            let p: Vec<&str> = line.split(',').collect();
            if p.len() < 6 { continue; }
            if let Some(&idx) = intel_idxs.get(card) {
                let g = &mut stats.gpus[idx];
                g.util  = p[1].trim().parse().unwrap_or(g.util);
                g.power = p[2].trim().parse().unwrap_or(g.power);
                g.temp  = p[3].trim().parse().unwrap_or(g.temp);
                if let Ok(b) = p[4].trim().parse::<u64>() { g.mem_used  = (b / 1_048_576) as u32; }
                if let Ok(b) = p[5].trim().parse::<u64>() { g.mem_total = (b / 1_048_576) as u32; }
                card += 1;
            }
        }
        return;
    }

    // Fallback: intel_gpu_top -J (one-shot sample — may not be available everywhere)
    let out = Command::new("intel_gpu_top")
        .args(["-J", "-s", "1"])
        .output();

    if let Ok(out) = out {
        let text = String::from_utf8_lossy(&out.stdout);
        if let Some(render_line) = text.lines().find(|l| l.contains("\"render\"") || l.contains("\"Render\"")) {
            if let Some(busy_pos) = render_line.find("\"busy\":") {
                let after = &render_line[busy_pos + 7..].trim_start();
                let end = after.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(after.len());
                if let Ok(util) = after[..end].parse::<f32>() {
                    for &idx in &intel_idxs {
                        stats.gpus[idx].util = util.round() as u32;
                    }
                }
            }
        }
    }
}

/// Minimal parser for rocm-smi `--json` output.
/// Returns `vec[(card_index, vec[(key_lowercase, value)])]`.
fn parse_rocm_json(json: &str) -> Vec<(usize, Vec<(String, String)>)> {
    let mut results = Vec::new();
    let mut remaining = json;

    while let Some(pos) = remaining.find("\"card") {
        remaining = &remaining[pos + 1..];
        let num_end = remaining.find('"').unwrap_or(remaining.len());
        let card_str = &remaining[..num_end];
        if !card_str.starts_with("card") { continue; }
        let card_num: usize = match card_str[4..].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        remaining = &remaining[num_end..];
        let brace_pos = match remaining.find('{') { Some(p) => p, None => break };
        remaining = &remaining[brace_pos + 1..];
        let close_pos = match remaining.find('}') { Some(p) => p, None => break };
        let block = &remaining[..close_pos];
        remaining = &remaining[close_pos + 1..];

        let mut pairs = Vec::new();
        let mut b = block;
        while let Some(ks) = b.find('"') {
            b = &b[ks + 1..];
            let ke = match b.find('"') { Some(p) => p, None => break };
            let key = b[..ke].to_ascii_lowercase();
            b = &b[ke + 1..];
            let vs = match b.find('"') { Some(p) => p, None => break };
            b = &b[vs + 1..];
            let ve = match b.find('"') { Some(p) => p, None => break };
            let val = b[..ve].to_string();
            b = &b[ve + 1..];
            pairs.push((key, val));
        }
        results.push((card_num, pairs));
    }
    results
}

pub fn run_tui(stats: Arc<Mutex<DashStats>>) {

    enable_raw_mode().unwrap();

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).unwrap();

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).unwrap();

    loop {

        if let Ok(mut s) = stats.lock() {

            update_gpu(&mut s);

            let total_power: f32 = s.gpus.iter().map(|g| g.power).sum();

            terminal.draw(|f| {

                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Length(3),
                        Constraint::Length(10),
                        Constraint::Min(5),
                    ])
                    .split(f.size());

                /* HEADER */

                let header = Line::from(vec![

                    Span::styled("⛏ XENOM MINER ", Style::default().add_modifier(Modifier::BOLD)),

                    Span::raw(" | 🌐 "),
                    Span::styled(s.rpcserver.clone(), Style::default().fg(Color::Cyan)),

                    Span::raw(" | "),
                    Span::styled("● Connected", Style::default().fg(Color::Green)),

                    Span::raw(" | 📊 "),
                    Span::styled(s.daa_score.to_string(), Style::default().fg(Color::Yellow)),

                    Span::raw(" | ⚙ "),
                    Span::styled(s.mode.clone(), Style::default().fg(Color::Green)),

                    Span::raw(" | 🧬 "),
                    Span::styled(
                        if s.genome_active { "Genome POW" } else { "KHeavyHash" },
                        Style::default().fg(Color::Magenta)
                    ),

                    Span::raw(" | ⏱ "),
                    Span::styled(s.uptime(), Style::default().fg(Color::Gray)),

                ]);

                let header_block = Block::default()
                    .title(" Status ")
                    .borders(Borders::ALL);

                f.render_widget(
                    Paragraph::new(header).block(header_block),
                    chunks[0],
                );

                /* PERFORMANCE */

                let perf = Line::from(vec![

                    Span::raw("⚡ Hashrate "),
                    Span::styled(fmt_hashrate(s.total_mhs), Style::default().fg(Color::Green)),

                    Span::raw("   📉 Diff "),
                    Span::styled(
                        format!("{:.2} MH", s.bits as f64 / 1_000_000.0),
                        Style::default().fg(Color::Yellow)
                    ),

                    Span::raw("   🏆 "),
                    Span::styled(s.accepted.to_string(), Style::default().fg(Color::Green)),

                    Span::raw("   ❌ "),
                    Span::styled(s.rejected.to_string(), Style::default().fg(Color::Red)),

                    Span::raw("   ⚡ "),
                    Span::styled(
                        format!("{:.0}W", total_power),
                        Style::default().fg(Color::Yellow)
                    ),

                ]);

                let perf_block = Block::default()
                    .title(" Performance ")
                    .borders(Borders::ALL);

                f.render_widget(
                    Paragraph::new(perf).block(perf_block),
                    chunks[1],
                );

                /* GPU TABLE */

                let rows: Vec<Row> = s.gpus.iter().map(|g| {

                    Row::new(vec![

                        Cell::from(g.id.to_string()),
                        Cell::from(g.name.clone()),
                        Cell::from(fmt_hashrate(g.hashrate)),

                        Cell::from(format!(
                            "{}°C {}",
                            g.temp,
                            bar(g.temp as f64,100.0,8)
                        ))
                        .style(Style::default().fg(temp_color(g.temp))),

                        Cell::from(format!(
                            "{}% {}",
                            g.fan,
                            bar(g.fan as f64,100.0,8)
                        )),

                        Cell::from(format!(
                            "{}% {}",
                            g.util,
                            bar(g.util as f64,100.0,8)
                        ))
                        .style(Style::default().fg(util_color(g.util))),

                        Cell::from(format!(
                            "{} / {} MB",
                            g.mem_used,
                            g.mem_total
                        )),

                        Cell::from(format!(
                            "{:.0}W {}",
                            g.power,
                            bar(g.power as f64,350.0,8)
                        ))
                        .style(Style::default().fg(power_color(g.power))),

                        Cell::from(g.accepted.to_string()),
                        Cell::from(g.rejected.to_string()),
                    ])

                }).collect();

                let table = Table::new(
                    rows,
                    [
                        Constraint::Length(3),
                        Constraint::Length(24),
                        Constraint::Length(10),
                        Constraint::Length(14),
                        Constraint::Length(14),
                        Constraint::Length(14),
                        Constraint::Length(16),
                        Constraint::Length(14),
                        Constraint::Length(6),
                        Constraint::Length(6),
                    ],
                )
                .header(
                    Row::new(vec![
                        "ID",
                        "🎮 GPU",
                        "Hashrate",
                        "🌡 Temp",
                        "🌀 Fan",
                        "⚙ Util",
                        "💾 Memory",
                        "⚡ Power",
                        "Acc",
                        "Rej",
                    ])
                    .style(Style::default().add_modifier(Modifier::BOLD))
                )
                .block(
                    Block::default()
                        .title(" GPU Workers ")
                        .borders(Borders::ALL)
                );

                f.render_widget(table, chunks[2]);

                /* LOGS */

                let logs: Vec<Line> = s
                    .log
                    .iter()
                    .rev()
                    .take(20)
                    .rev()
                    .map(|l| Line::from(l.clone()))
                    .collect();

                let log_block = Block::default()
                    .title(" Logs (q quit) ")
                    .borders(Borders::ALL);

                f.render_widget(
                    Paragraph::new(logs).block(log_block),
                    chunks[3],
                );

            }).unwrap();
        }

        if event::poll(Duration::from_millis(200)).unwrap() {

            if let Event::Key(key) = event::read().unwrap() {

                if key.code == KeyCode::Char('q')
                    || (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    break;
                }
            }
        }
    }

    disable_raw_mode().unwrap();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).unwrap();
}
