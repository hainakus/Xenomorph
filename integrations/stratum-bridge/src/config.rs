use serde::{Deserialize, Serialize};
use std::path::Path;

// ── Bridge config (TOML) ──────────────────────────────────────────────────────

/// Top-level configuration loaded from `--config <file.toml>`.
/// All fields are optional; CLI flags override TOML values.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct BridgeConfig {
    #[serde(default)]
    pub bridge: BridgeSection,
    #[serde(default)]
    pub node: NodeSection,
    #[serde(default)]
    pub l2: L2Section,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct BridgeSection {
    /// Pool theme: "genetics" | "climate" | "ai" | "materials" | "generic"
    #[serde(default)]
    pub theme: String,
    /// Human-readable pool name (shown in API)
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct NodeSection {
    /// Xenom node gRPC endpoint (host:port, no scheme)
    #[serde(default)]
    pub rpc: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct L2Section {
    /// Enable L2 job dispatch alongside PoW jobs.
    #[serde(default)]
    pub enabled: bool,
    /// Base URL of the genetics/climate/ai coordinator REST API.
    #[serde(default)]
    pub coordinator: String,
    /// Dataset identifier sent to miners (informational).
    #[serde(default)]
    pub dataset: String,
    /// Poll interval in seconds for fetching new L2 jobs.
    #[serde(default = "default_poll_secs")]
    pub poll_secs: u64,
    /// Maximum number of workers allowed to claim the same L2 job simultaneously.
    #[serde(default = "default_l2_workers")]
    pub max_workers: usize,
}

fn default_poll_secs() -> u64 { 10 }
fn default_l2_workers() -> usize { 4 }

impl BridgeConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read config '{}': {e}", path.display()))?;
        toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("TOML parse error in '{}': {e}", path.display()))
    }

    pub fn theme(&self) -> &str {
        if self.bridge.theme.is_empty() { "generic" } else { &self.bridge.theme }
    }
}
