use std::{
    collections::VecDeque,
    io,
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
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
    Frame, Terminal,
};

const LOG_CAP: usize = 200;
const LOG_DISPLAY: usize = 25;

// ── Shared state ──────────────────────────────────────────────────────────────

pub struct GpuStats {
    pub id:       usize,
    pub name:     String,
    pub hashrate: f64,   // MH/s
    pub accepted: u64,
    pub rejected: u64,
}

pub struct DashStats {
    pub rpcserver:     String,
    pub connected:     bool,
    pub mode:          String,
    pub daa_score:     u64,
    pub bits:          u32,
    pub genome_active: bool,
    pub total_mhs:     f64,
    pub accepted:      u64,
    pub rejected:      u64,
    pub gpus:          Vec<GpuStats>,
    pub num_cpus:      usize,
    pub log:           VecDeque<String>,
    start:             Instant,
}

impl DashStats {
    pub fn new(rpcserver: String, mode: String, num_cpus: usize) -> Self {
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
            num_cpus,
            log: VecDeque::with_capacity(LOG_CAP),
            start: Instant::now(),
        }
    }

    pub fn push_log(&mut self, msg: impl Into<String>) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let h = (secs / 3600) % 24;
        let m = (secs / 60) % 60;
        let s = secs % 60;
        let line = format!("{h:02}:{m:02}:{s:02}  {}", msg.into());
        if self.log.len() >= LOG_CAP {
            self.log.pop_front();
        }
        self.log.push_back(line);
    }

    fn uptime(&self) -> String {
        let e = self.start.elapsed().as_secs();
        format!("{:02}:{:02}:{:02}", e / 3600, (e / 60) % 60, e % 60)
    }
}

// ── TUI entry point ───────────────────────────────────────────────────────────

/// Runs the ratatui dashboard in the current thread.
/// Returns when the user presses `q` or `Ctrl+C`, then calls `process::exit(0)`.
/// Falls back silently if the terminal does not support raw mode (e.g. piped output).
pub fn run_tui(stats: Arc<Mutex<DashStats>>) {
    if enable_raw_mode().is_err() {
        return;
    }
    let mut stdout = io::stdout();
    if execute!(stdout, EnterAlternateScreen).is_err() {
        let _ = disable_raw_mode();
        return;
    }

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(_) => {
            let _ = disable_raw_mode();
            return;
        }
    };

    loop {
        if let Ok(s) = stats.lock() {
            let _ = terminal.draw(|f| draw(f, &s));
        }

        if event::poll(Duration::from_millis(150)).unwrap_or(false) {
            if let Ok(Event::Key(k)) = event::read() {
                let quit = k.code == KeyCode::Char('q')
                    || k.code == KeyCode::Char('Q')
                    || (k.code == KeyCode::Char('c')
                        && k.modifiers.contains(KeyModifiers::CONTROL));
                if quit {
                    break;
                }
            }
        }
    }

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    std::process::exit(0);
}

// ── Layout ───────────────────────────────────────────────────────────────────

fn draw(f: &mut Frame, s: &DashStats) {
    let worker_rows = if s.num_cpus > 0 { 1 } else { s.gpus.len().max(1) };
    let workers_h   = (worker_rows + 4) as u16; // 2 borders + 1 header + data rows

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Length(workers_h),
            Constraint::Length((LOG_DISPLAY + 2) as u16), // 25 lines + 2 borders, fixed
        ])
        .split(f.size());

    render_header(f, chunks[0], s);
    render_perf(f, chunks[1], s);
    render_workers(f, chunks[2], s);
    render_log(f, chunks[3], s);
}

// ── Header bar ───────────────────────────────────────────────────────────────

fn render_header(f: &mut Frame, area: ratatui::layout::Rect, s: &DashStats) {
    let conn_style = if s.connected {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::Red)
    };
    let conn_label = if s.connected { "● Connected" } else { "○ Disconnected" };
    let mode_color = if s.genome_active { Color::Cyan } else { Color::Yellow };

    let line = Line::from(vec![
        Span::styled(" grpc://", Style::default().fg(Color::DarkGray)),
        Span::styled(s.rpcserver.clone(), Style::default().fg(Color::White)),
        Span::raw("  "),
        Span::styled(conn_label, conn_style),
        Span::raw("  │  DAA: "),
        Span::styled(
            format!("{}", s.daa_score),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  │  Mode: "),
        Span::styled(s.mode.clone(), Style::default().fg(mode_color)),
        Span::raw("  │  Up: "),
        Span::styled(s.uptime(), Style::default().fg(Color::DarkGray)),
        Span::raw(" "),
    ]);

    let block = Block::default()
        .title(Span::styled(
            " ⛏  XENOM MINER ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    f.render_widget(Paragraph::new(line).block(block), area);
}

// ── Performance stats ────────────────────────────────────────────────────────

fn render_perf(f: &mut Frame, area: ratatui::layout::Rect, s: &DashStats) {
    let diff = format_difficulty(s.bits);
    let mhs  = fmt_mhs(s.total_mhs);

    let line1 = Line::from(vec![
        Span::raw("  Hashrate: "),
        Span::styled(
            mhs,
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("   Difficulty: "),
        Span::styled(diff, Style::default().fg(Color::Yellow)),
    ]);

    let rej_color = if s.rejected > 0 { Color::Red } else { Color::DarkGray };
    let line2 = Line::from(vec![
        Span::raw("  Accepted: "),
        Span::styled(
            s.accepted.to_string(),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("   Rejected: "),
        Span::styled(s.rejected.to_string(), Style::default().fg(rej_color)),
    ]);

    let block = Block::default()
        .title(" Performance ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));

    f.render_widget(Paragraph::new(vec![line1, line2]).block(block), area);
}

// ── Workers table ────────────────────────────────────────────────────────────

fn render_workers(f: &mut Frame, area: ratatui::layout::Rect, s: &DashStats) {
    let title = if s.num_cpus > 0 {
        format!(" CPU Workers ({} threads) ", s.num_cpus)
    } else {
        format!(" GPU Workers ({}) ", s.gpus.len())
    };

    let hdr_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);

    let header = Row::new(vec![
        Cell::from("#").style(hdr_style),
        Cell::from("Name").style(hdr_style),
        Cell::from("Hashrate").style(hdr_style),
        Cell::from("Accepted").style(hdr_style),
        Cell::from("Rejected").style(hdr_style),
    ]);

    let rows: Vec<Row> = if s.num_cpus > 0 {
        vec![Row::new(vec![
            Cell::from("CPU"),
            Cell::from(format!("{} threads", s.num_cpus)),
            Cell::from(fmt_mhs(s.total_mhs)).style(Style::default().fg(Color::Green)),
            Cell::from(s.accepted.to_string()).style(Style::default().fg(Color::Green)),
            Cell::from(s.rejected.to_string()),
        ])]
    } else if s.gpus.is_empty() {
        vec![Row::new(vec![
            Cell::from("—"),
            Cell::from("Initialising..."),
            Cell::from(""),
            Cell::from(""),
            Cell::from(""),
        ])]
    } else {
        s.gpus
            .iter()
            .map(|g| {
                let rej_sty = if g.rejected > 0 {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                Row::new(vec![
                    Cell::from(g.id.to_string()),
                    Cell::from(g.name.clone()),
                    Cell::from(fmt_mhs(g.hashrate))
                        .style(Style::default().fg(Color::Green)),
                    Cell::from(g.accepted.to_string())
                        .style(Style::default().fg(Color::Green)),
                    Cell::from(g.rejected.to_string()).style(rej_sty),
                ])
            })
            .collect()
    };

    let widths = [
        Constraint::Length(4),
        Constraint::Min(22),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Length(10),
    ];

    let table = Table::new(rows, widths)
        .header(header.bottom_margin(0))
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        )
        .highlight_style(Style::default().add_modifier(Modifier::BOLD));

    f.render_stateful_widget(table, area, &mut TableState::default());
}

// ── Log pane ─────────────────────────────────────────────────────────────────

fn render_log(f: &mut Frame, area: ratatui::layout::Rect, s: &DashStats) {
    let lines: Vec<Line> = s
        .log
        .iter()
        .rev()
        .take(LOG_DISPLAY)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|l| {
            let style = if l.contains("Success") || l.contains("PASS") {
                Style::default().fg(Color::Green)
            } else if l.contains("Reject")
                || l.contains("false-positive")
                || l.contains("WARN")
                || l.contains("error")
            {
                Style::default().fg(Color::Red)
            } else if l.contains("template") || l.contains("daa=") {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default().fg(Color::White)
            };
            Line::from(Span::styled(format!("  {l}"), style))
        })
        .collect();

    let block = Block::default()
        .title(" Log   [q] quit ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));

    f.render_widget(Paragraph::new(lines).block(block), area);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn fmt_mhs(mhs: f64) -> String {
    if mhs >= 1_000.0 {
        format!("{:.3} GH/s", mhs / 1_000.0)
    } else if mhs >= 1.0 {
        format!("{:.2} MH/s", mhs)
    } else if mhs > 0.0 {
        format!("{:.2} KH/s", mhs * 1_000.0)
    } else {
        "  0 H/s".to_owned()
    }
}

pub fn format_difficulty(bits: u32) -> String {
    if bits == 0 {
        return "—".to_owned();
    }
    // Approximate log2(difficulty) from compact exponent.
    // compact: mantissa * 2^(8*(exp-3)), difficulty ≈ 2^(256 - 8*exp)
    let exp = (bits >> 24) as i32;
    let diff_log2 = (256 - 8 * exp).max(0) as f64;
    let diff = 2f64.powf(diff_log2);
    if diff >= 1e18 {
        format!("{:.2} EH", diff / 1e18)
    } else if diff >= 1e15 {
        format!("{:.2} PH", diff / 1e15)
    } else if diff >= 1e12 {
        format!("{:.2} TH", diff / 1e12)
    } else if diff >= 1e9 {
        format!("{:.2} GH", diff / 1e9)
    } else if diff >= 1e6 {
        format!("{:.2} MH", diff / 1e6)
    } else {
        format!("{:.0} H", diff)
    }
}
