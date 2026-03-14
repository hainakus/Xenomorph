use anyhow::{Context, Result};
use bioproof_core::{compute_proof, merkle_root, ArtifactType, OutputEntry};
use std::path::Path;
use std::str::FromStr;

/// Collect all file paths under `dir` recursively, sorted for determinism.
pub async fn collect_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let mut rd = tokio::fs::read_dir(&current).await?;
        while let Some(entry) = rd.next_entry().await? {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                files.push(p);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// Compute the combined BLAKE3 Merkle root over all files in a directory.
/// Deterministic: files are sorted before hashing.
pub async fn hash_directory(dir: &Path) -> Result<String> {
    let entries = collect_files(dir).await?;
    if entries.is_empty() {
        return Ok(hex::encode([0u8; 32]));
    }
    let mut leaf_hashes = Vec::with_capacity(entries.len());
    for path in &entries {
        let data = tokio::fs::read(path)
            .await
            .with_context(|| format!("cannot read {}", path.display()))?;
        let (root_hex, _, _) = compute_proof(&data, 0);
        let mut b = [0u8; 32];
        if let Ok(v) = hex::decode(&root_hex) {
            if let Ok(arr) = v.try_into() {
                b = arr;
            }
        }
        leaf_hashes.push(b);
    }
    Ok(hex::encode(merkle_root(&leaf_hashes)))
}

/// Hash each output file individually and return a list of `OutputEntry`.
pub async fn hash_output_files(dir: &Path) -> Result<Vec<OutputEntry>> {
    let files = collect_files(dir).await?;
    let mut entries = Vec::with_capacity(files.len());
    for path in &files {
        let data = tokio::fs::read(path)
            .await
            .with_context(|| format!("cannot read {}", path.display()))?;
        let size_bytes = data.len() as u64;
        let (proof_root, file_hash, _) = compute_proof(&data, 0);
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let kind = path
            .extension()
            .and_then(|e| e.to_str())
            .and_then(|ext| ArtifactType::from_str(ext).ok())
            .unwrap_or_else(|| ArtifactType::Other("bin".to_owned()));

        entries.push(OutputEntry { name, kind, proof_root, file_hash, size_bytes });
    }
    Ok(entries)
}

/// Combined Merkle root over all OutputEntry proof_roots.
pub fn combined_output_root(outputs: &[OutputEntry]) -> String {
    let leaves: Vec<[u8; 32]> = outputs
        .iter()
        .map(|o| {
            let mut b = [0u8; 32];
            if let Ok(v) = hex::decode(&o.proof_root) {
                if let Ok(arr) = v.try_into() {
                    b = arr;
                }
            }
            b
        })
        .collect();
    hex::encode(merkle_root(&leaves))
}
