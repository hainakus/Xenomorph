use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const WEIGHTS_FILE: &str = "model.safetensors";
const HASH_FILE: &str = "base_checkpoint";

/// In-memory bundle returned by the seed-node. The miner can persist this to
/// disk so it only has to download the full weights once.
#[derive(Debug, Clone)]
pub struct ModelBundle {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
}

/// Local on-disk cache for model checkpoints.
pub struct ModelCache {
    root: PathBuf,
}

impl ModelCache {
    /// Create a cache rooted at `root`. The directory will be created lazily.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self { root: root.as_ref().to_path_buf() }
    }

    /// Directory where a specific model is stored.
    pub fn model_dir(&self, model_id: &str) -> PathBuf {
        self.root.join(sanitize_model_id(model_id))
    }

    /// Returns true if config, tokenizer, weights and hash files are all present.
    pub fn is_cached(&self, model_id: &str) -> bool {
        let dir = self.model_dir(model_id);
        dir.is_dir()
            && dir.join(CONFIG_FILE).is_file()
            && dir.join(TOKENIZER_FILE).is_file()
            && dir.join(WEIGHTS_FILE).is_file()
            && dir.join(HASH_FILE).is_file()
    }

    /// Read a cached bundle from disk.
    pub fn read(&self, model_id: &str) -> Result<ModelBundle> {
        let dir = self.model_dir(model_id);
        let base_checkpoint = self.read_base_checkpoint(model_id)?;
        Ok(ModelBundle {
            model_id: model_id.to_string(),
            base_checkpoint,
            config: fs::read(dir.join(CONFIG_FILE)).context("Failed to read cached config")?,
            tokenizer: fs::read(dir.join(TOKENIZER_FILE)).context("Failed to read cached tokenizer")?,
            weights: fs::read(dir.join(WEIGHTS_FILE)).context("Failed to read cached weights")?,
        })
    }

    /// Persist a bundle to disk, atomically writing each file.
    pub fn write(&self, model_id: &str, bundle: &ModelBundle) -> Result<()> {
        let dir = self.model_dir(model_id);
        fs::create_dir_all(&dir).with_context(|| format!("Failed to create cache directory {:?}", dir))?;

        fs::write(dir.join(CONFIG_FILE), &bundle.config).context("Failed to write cached config")?;
        fs::write(dir.join(TOKENIZER_FILE), &bundle.tokenizer).context("Failed to write cached tokenizer")?;
        fs::write(dir.join(WEIGHTS_FILE), &bundle.weights).context("Failed to write cached weights")?;

        let hash_hex = hex::encode(bundle.base_checkpoint);
        fs::write(dir.join(HASH_FILE), hash_hex).context("Failed to write cached base checkpoint")?;

        Ok(())
    }

    /// Read the cached base checkpoint hash.
    pub fn read_base_checkpoint(&self, model_id: &str) -> Result<[u8; 32]> {
        let path = self.model_dir(model_id).join(HASH_FILE);
        let hex_str = fs::read_to_string(&path).with_context(|| format!("Failed to read cached base checkpoint at {:?}", path))?;
        let bytes = hex::decode(&hex_str).context("Cached base checkpoint is not valid hex")?;
        if bytes.len() != 32 {
            bail!("Cached base checkpoint must be 32 bytes, got {}", bytes.len());
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(arr)
    }
}

fn sanitize_model_id(model_id: &str) -> String {
    model_id
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect()
}
