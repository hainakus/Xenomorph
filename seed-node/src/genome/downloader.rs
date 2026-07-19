use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use futures::StreamExt;
use reqwest;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::info;

use super::archive::{GenomeArchive, DEFAULT_FRAGMENT_SIZE};

/// Downloads `.xenom` genome archives over HTTP or via an IPFS gateway.
pub struct GenomeDownloader {
    ipfs_gateway: String,
    http_client: reqwest::Client,
}

impl Default for GenomeDownloader {
    fn default() -> Self {
        Self::new("https://ipfs.io/ipfs".to_string())
    }
}

impl GenomeDownloader {
    /// Create a downloader with a specific IPFS gateway base URL.
    pub fn new(ipfs_gateway: String) -> Self {
        Self {
            ipfs_gateway,
            http_client: reqwest::Client::new(),
        }
    }

    /// Download a genome archive to `dest`.
    ///
    /// `source` may be:
    /// - An HTTP(S) URL (`http://...` or `https://...`)
    /// - An `ipfs://<cid>` URI
    /// - A raw CID or hash, which is resolved through the configured IPFS gateway
    pub async fn download(&self, source: &str, dest: &Path) -> Result<()> {
        let url = self.resolve_url(source)?;

        info!("Downloading genome archive from {} to {:?}", url, dest);

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).await?;
        }

        let response = self
            .http_client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("Failed to fetch genome archive from {}", url))?;

        if !response.status().is_success() {
            bail!("HTTP {} when downloading genome archive from {}", response.status(), url);
        }

        let mut file = fs::File::create(dest).await?;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error while downloading genome archive")?;
            file.write_all(&chunk).await?;
        }

        file.flush().await?;
        info!("Downloaded genome archive to {:?}", dest);
        Ok(())
    }

    /// Download a genome archive to `dest` only if the local copy is missing, then load
    /// and verify it against the expected merkle root.
    pub async fn get_or_download(
        &self,
        source: &str,
        dest: &Path,
        expected_merkle: [u8; 32],
    ) -> Result<GenomeArchive> {
        if !dest.exists() {
            self.download(source, dest).await?;
        } else {
            info!("Using cached genome archive at {:?}", dest);
        }
        self.verify_and_load(dest, expected_merkle).await
    }

    /// Load a genome archive from disk and verify its merkle root.
    pub async fn verify_and_load<P: AsRef<Path>>(
        &self,
        path: P,
        expected_merkle: [u8; 32],
    ) -> Result<GenomeArchive> {
        let path = path.as_ref();
        let archive = GenomeArchive::load_with_fragment_size(path, DEFAULT_FRAGMENT_SIZE)
            .with_context(|| format!("Failed to load genome archive from {:?}", path))?;

        if archive.header.merkle_root != expected_merkle {
            bail!(
                "Genome merkle root mismatch: expected {}, got {}",
                hex::encode(expected_merkle),
                hex::encode(archive.header.merkle_root)
            );
        }

        if !archive.verify_merkle() {
            bail!("Genome archive merkle verification failed for {:?}", path);
        }

        info!("Verified genome archive {:?} with merkle {}", path, hex::encode(expected_merkle));
        Ok(archive)
    }

    /// Return the local cache path for a given merkle root.
    pub fn cache_path_for<P: AsRef<Path>>(cache_dir: P, merkle_root: &[u8; 32]) -> PathBuf {
        cache_dir.as_ref().join(hex::encode(merkle_root)).join("genome.xenom")
    }

    fn resolve_url(&self, source: &str) -> Result<String> {
        if source.starts_with("http://") || source.starts_with("https://") {
            Ok(source.to_string())
        } else if let Some(stripped) = source.strip_prefix("ipfs://") {
            Ok(format!("{}/{}", self.ipfs_gateway.trim_end_matches('/'), stripped))
        } else if source.contains('/') || source.contains(':') {
            bail!("Unsupported genome source: {}", source)
        } else {
            // Treat as a raw CID.
            Ok(format!("{}/{}", self.ipfs_gateway.trim_end_matches('/'), source))
        }
    }
}
