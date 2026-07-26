use std::sync::Arc;

use anyhow::{Context, Result};
use xenom_quic::{async_trait, CheckpointFileRequest, CheckpointFileType, FileProvider};

use crate::model::manager::ModelManager;

/// QUIC file provider backed by the local `ModelManager`.
///
/// Serves encrypted checkpoint bytes directly from disk. `Config`, `Tokenizer`
/// and `Weights` are served from the active checkpoint; `Adapter` returns the
/// encrypted LoRA adapter bytes when the requested base matches the active base.
pub struct ModelFileProvider {
    model_manager: Arc<ModelManager>,
}

impl ModelFileProvider {
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self { model_manager }
    }
}

#[async_trait]
impl FileProvider for ModelFileProvider {
    async fn get_file(&self, request: &CheckpointFileRequest) -> Result<Option<(Vec<u8>, [u8; 32])>> {
        // Refuse to serve anything other than the active checkpoint for this model.
        let active_hash = self.model_manager.active_hash(&request.model_id).await;
        if active_hash != Some(request.weights_hash) {
            return Ok(None);
        }

        let (head_hash, base_hash) =
            self.model_manager.get_model_checkpoint_info_v2(&request.model_id).await.context("failed to load checkpoint metadata")?;

        if head_hash != request.weights_hash {
            return Ok(None);
        }

        // Adapter-only request: serve the LoRA adapter bytes.
        let cached_base_hash = if request.file_type == CheckpointFileType::Adapter { Some(base_hash) } else { None };

        let (files, ..) = self
            .model_manager
            .get_encrypted_model_checkpoint_v2(&request.model_id, cached_base_hash)
            .await
            .context("failed to load encrypted model files")?;

        let bytes = match request.file_type {
            CheckpointFileType::Config => files.config,
            CheckpointFileType::Tokenizer => files.tokenizer,
            CheckpointFileType::Weights | CheckpointFileType::Adapter => files.weights,
        };

        let hash = blake3::hash(&bytes);
        Ok(Some((bytes, *hash.as_bytes())))
    }
}
