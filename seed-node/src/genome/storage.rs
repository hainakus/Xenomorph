use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use tokio::fs;
use tracing::{info, warn};

use super::archive::GenomeArchive;
use super::downloader::GenomeDownloader;

/// Cache and manage `.xenom` genome archives.
pub struct GenomeStorage {
    cache_dir: PathBuf,
    active_genomes: HashMap<[u8; 32], Arc<GenomeArchive>>,
    downloader: GenomeDownloader,
}

impl GenomeStorage {
    /// Create a new storage instance rooted at `cache_dir`.
    pub async fn new<P: AsRef<Path>>(cache_dir: P) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir).await?;
        Ok(Self {
            cache_dir,
            active_genomes: HashMap::new(),
            downloader: GenomeDownloader::default(),
        })
    }

    /// Create a new storage instance with a custom downloader.
    pub async fn new_with_downloader<P: AsRef<Path>>(
        cache_dir: P,
        downloader: GenomeDownloader,
    ) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir).await?;
        Ok(Self {
            cache_dir,
            active_genomes: HashMap::new(),
            downloader,
        })
    }

    /// Return a loaded genome archive, fetching it from `source` if it is not already cached.
    pub async fn get_or_load(
        &mut self,
        merkle_root: [u8; 32],
        source: &str,
    ) -> Result<Arc<GenomeArchive>> {
        if let Some(archive) = self.active_genomes.get(&merkle_root) {
            return Ok(archive.clone());
        }

        let cache_path = GenomeDownloader::cache_path_for(&self.cache_dir, &merkle_root);
        let archive = self
            .downloader
            .get_or_download(source, &cache_path, merkle_root)
            .await
            .with_context(|| {
                format!(
                    "Failed to obtain genome archive for merkle {} from {}",
                    hex::encode(merkle_root),
                    source
                )
            })?;

        let archive = Arc::new(archive);
        self.active_genomes.insert(merkle_root, archive.clone());
        info!("Loaded genome archive {} into memory", hex::encode(merkle_root));
        Ok(archive)
    }

    /// Load a genome archive from a local path and cache it under `merkle_root`.
    pub async fn load_from_path<P: AsRef<Path>>(
        &mut self,
        merkle_root: [u8; 32],
        path: P,
    ) -> Result<Arc<GenomeArchive>> {
        if let Some(archive) = self.active_genomes.get(&merkle_root) {
            return Ok(archive.clone());
        }

        let archive = GenomeArchive::load(path.as_ref())
            .with_context(|| format!("Failed to load genome archive from {:?}", path.as_ref()))?;

        if archive.header.merkle_root != merkle_root {
            bail!(
                "Loaded genome merkle root mismatch: expected {}, got {}",
                hex::encode(merkle_root),
                hex::encode(archive.header.merkle_root)
            );
        }

        if !archive.verify_merkle() {
            bail!("Genome archive merkle verification failed for {:?}", path.as_ref());
        }

        let archive = Arc::new(archive);
        self.active_genomes.insert(merkle_root, archive.clone());
        Ok(archive)
    }

    /// Return the list of merkle roots currently held in memory.
    pub fn list_available(&self) -> Vec<[u8; 32]> {
        self.active_genomes.keys().copied().collect()
    }

    /// Remove a genome from the in-memory cache.
    pub fn unload(&mut self, merkle_root: &[u8; 32]) {
        if self.active_genomes.remove(merkle_root).is_some() {
            warn!("Unloaded genome archive {}", hex::encode(merkle_root));
        }
    }
}
