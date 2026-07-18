use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::info;

const DEFAULT_RPC_URL: &str = "ws://localhost:16110";
const DEFAULT_MODEL_ID: &str = "dnabert2";
const DEFAULT_THREADS: usize = 4;
const DEFAULT_DATA_DIR_NAME: &str = ".xenom-miner";

/// Top-level miner configuration that can be persisted to disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerConfig {
    pub wallet_address: String,
    pub rpc_url: String,
    pub model_id: String,
    pub threads: usize,
    pub mock_mode: bool,
    pub dry_run: bool,
    pub data_dir: PathBuf,
}

impl Default for MinerConfig {
    fn default() -> Self {
        Self {
            wallet_address: String::new(),
            rpc_url: DEFAULT_RPC_URL.to_string(),
            model_id: DEFAULT_MODEL_ID.to_string(),
            threads: DEFAULT_THREADS,
            mock_mode: false,
            dry_run: false,
            data_dir: default_data_dir(),
        }
    }
}

impl MinerConfig {
    /// Load configuration from the default location inside `data_dir`,
    /// creating a fresh one when it does not exist.
    pub fn load_or_create(data_dir: &Path) -> Result<Self> {
        let path = config_path(data_dir);

        if path.exists() {
            let bytes = std::fs::read(&path).with_context(|| format!("Failed to read config at {:?}", path))?;
            let config: MinerConfig = serde_json::from_slice(&bytes).with_context(|| "Failed to parse config as JSON")?;
            info!("Loaded miner config from {:?}", path);
            Ok(config)
        } else {
            let config = MinerConfig { data_dir: data_dir.to_path_buf(), ..MinerConfig::default() };
            config.save(data_dir)?;
            info!("Created new miner config at {:?}", path);
            Ok(config)
        }
    }

    /// Save configuration to disk.
    pub fn save(&self, data_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(data_dir).with_context(|| format!("Failed to create data directory {:?}", data_dir))?;
        let path = config_path(data_dir);
        let bytes = serde_json::to_vec_pretty(self).with_context(|| "Failed to serialize config")?;
        std::fs::write(&path, bytes).with_context(|| format!("Failed to write config to {:?}", path))?;
        Ok(())
    }
}

/// Return the default data directory in the user's home folder.
pub fn default_data_dir() -> PathBuf {
    dirs::home_dir().map(|h| h.join(DEFAULT_DATA_DIR_NAME)).unwrap_or_else(|| PathBuf::from(DEFAULT_DATA_DIR_NAME))
}

fn config_path(data_dir: &Path) -> PathBuf {
    data_dir.join("config.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_default_config() {
        let cfg = MinerConfig::default();
        assert_eq!(cfg.rpc_url, DEFAULT_RPC_URL);
        assert_eq!(cfg.model_id, DEFAULT_MODEL_ID);
        assert_eq!(cfg.threads, DEFAULT_THREADS);
    }

    #[test]
    fn test_save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = MinerConfig::default();
        cfg.data_dir = tmp.path().to_path_buf();
        cfg.wallet_address = "xnom:test".to_string();

        cfg.save(tmp.path()).unwrap();
        let loaded = MinerConfig::load_or_create(tmp.path()).unwrap();
        assert_eq!(loaded.wallet_address, "xnom:test");
        assert_eq!(loaded.model_id, cfg.model_id);
    }

    #[test]
    fn test_load_invalid_config_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let path = config_path(tmp.path());
        std::fs::create_dir_all(tmp.path()).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"not json").unwrap();

        assert!(MinerConfig::load_or_create(tmp.path()).is_err());
    }
}
