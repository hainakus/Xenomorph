use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

use crate::rpc::messages::{TrainingBatch};

use super::checkpoint::{ModelCheckpoint, ModelMetrics};
use super::downloader::download_model;
use super::storage::ModelStorage;
use super::RawModelFiles;

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: u32,
    pub category: String,
    pub checkpoint: ModelCheckpoint,
    pub loaded: bool,
    pub last_used: u64,
}

pub struct ModelManager {
    base_path: String,
    storage: Arc<ModelStorage>,
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    node_id: String,
}

impl ModelManager {
    pub async fn new(base_path: String) -> Result<Self> {
        let key = ModelStorage::generate_key();
        let storage = Arc::new(ModelStorage::new(base_path.clone(), key));

        // Create base directory if it doesn't exist
        tokio::fs::create_dir_all(&base_path).await?;

        let node_id = Uuid::new_v4().to_string();

        Ok(Self { base_path, storage, models: Arc::new(RwLock::new(HashMap::new())), node_id })
    }

    pub async fn load_model(&self, model_id: &str) -> Result<ModelInfo> {
        // Check if already loaded
        {
            let models = self.models.read().await;
            if let Some(model) = models.get(model_id) {
                return Ok(model.clone());
            }
        }

        // Load from storage. New checkpoints store config/tokenizer/weights; legacy ones use a single model.enc.
        let data = match self.storage.load_model_files(model_id).await {
            Ok(files) => files.weights,
            Err(_) => self.storage.load_model(model_id).await.map_err(|e| anyhow!("Failed to load model: {}", e))?,
        };

        // Parse checkpoint from data (simplified - in production would deserialize)
        let checkpoint = ModelCheckpoint::new(0, model_id.to_string(), 1, &data, ModelMetrics::default());

        let model_info = ModelInfo {
            id: model_id.to_string(),
            name: model_id.to_string(),
            version: 1,
            category: "NLP".to_string(),
            checkpoint,
            loaded: true,
            last_used: chrono::Utc::now().timestamp() as u64,
        };

        // Cache the model
        {
            let mut models = self.models.write().await;
            models.insert(model_id.to_string(), model_info.clone());
        }

        info!("Loaded model: {}", model_id);
        Ok(model_info)
    }

    /// Ensure a model is available locally, downloading it from Hugging Face if needed.
    /// Does nothing if the model is already stored.
    pub async fn ensure_model_downloaded(&self, model_id: &str) -> Result<()> {
        // Check if a model file or checkpoint already exists for this id.
        if self.storage.model_exists(model_id).await {
            info!("Model {} already exists locally; skipping download", model_id);
            return Ok(());
        }

        info!("Model {} not found locally; downloading from Hugging Face", model_id);
        let files = download_model(model_id).await?;

        let metrics = ModelMetrics::default();
        self.store_model_files(model_id, &files, metrics).await?;

        info!("Downloaded and stored model {} (weights {} bytes)", model_id, files.weights.len());
        Ok(())
    }

    pub async fn store_model(&self, model_id: &str, data: &[u8], metrics: ModelMetrics) -> Result<String> {
        let checkpoint = ModelCheckpoint::new(0, model_id.to_string(), 1, data, metrics);

        let path = self.storage.store_model(model_id, data).await.map_err(|e| anyhow!("Failed to store model: {}", e))?;

        let model_info = ModelInfo {
            id: model_id.to_string(),
            name: model_id.to_string(),
            version: 1,
            category: "NLP".to_string(),
            checkpoint,
            loaded: false,
            last_used: 0,
        };

        {
            let mut models = self.models.write().await;
            models.insert(model_id.to_string(), model_info);
        }

        info!("Stored model: {} at {}", model_id, path);
        Ok(path)
    }

    pub async fn store_model_files(&self, model_id: &str, files: &RawModelFiles, metrics: ModelMetrics) -> Result<()> {
        self.storage.store_model_files(model_id, files).await.map_err(|e| anyhow!("Failed to store model files: {}", e))?;

        let checkpoint = ModelCheckpoint::new(0, model_id.to_string(), 1, &files.weights, metrics);

        let model_info = ModelInfo {
            id: model_id.to_string(),
            name: model_id.to_string(),
            version: 1,
            category: "NLP".to_string(),
            checkpoint,
            loaded: false,
            last_used: 0,
        };

        {
            let mut models = self.models.write().await;
            models.insert(model_id.to_string(), model_info);
        }

        info!("Stored model files for {} (weights {} bytes)", model_id, files.weights.len());
        Ok(())
    }

    /// Build a `TrainingBatch` that ties the model checkpoint to an optional genome merkle root.
    pub async fn get_training_batch_with_genome(
        &self,
        model_id: &str,
        genome_root: Option<[u8; 32]>,
    ) -> Result<TrainingBatch> {
        let base_checkpoint = if let Some(root) = genome_root {
            root
        } else {
            // Fall back to the current model checkpoint hash if no genome root is provided.
            match self.get_model(model_id).await {
                Some(info) => info.checkpoint.weights_hash,
                None => [0u8; 32],
            }
        };

        Ok(TrainingBatch {
            batch_id: 1,
            model_id: model_id.to_string(),
            base_checkpoint,
            data_indices: vec![],
            target_improvement: 0.01,
            learning_rate: 0.01,
        })
    }

    /// Return the raw model checkpoint files (config, tokenizer, weights) for a model id.
    pub async fn get_model_checkpoint(&self, model_id: &str) -> Result<(ModelCheckpoint, RawModelFiles)> {
        let files = self.storage.load_model_files(model_id).await.map_err(|e| anyhow!("Failed to load model files: {}", e))?;
        let checkpoint = ModelCheckpoint::new(0, model_id.to_string(), 1, &files.weights, ModelMetrics::default());
        Ok((checkpoint, files))
    }

    pub async fn store_checkpoint(&self, model_id: &str, version: u32, data: &[u8], _metrics: ModelMetrics) -> Result<String> {
        let path =
            self.storage.store_checkpoint(model_id, version, data).await.map_err(|e| anyhow!("Failed to store checkpoint: {}", e))?;

        info!("Stored checkpoint: {} v{} at {}", model_id, version, path);
        Ok(path)
    }

    pub async fn load_checkpoint(&self, model_id: &str, version: u32) -> Result<Vec<u8>> {
        self.storage.load_checkpoint(model_id, version).await.map_err(|e| anyhow!("Failed to load checkpoint: {}", e))
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let models = self.models.read().await;
        Ok(models.values().cloned().collect())
    }

    pub async fn get_model(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }

    pub async fn update_last_used(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write().await;
        if let Some(model) = models.get_mut(model_id) {
            model.last_used = chrono::Utc::now().timestamp() as u64;
        }
        Ok(())
    }

    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write().await;
        if let Some(mut model) = models.remove(model_id) {
            model.loaded = false;
            info!("Unloaded model: {}", model_id);
        }
        Ok(())
    }

    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        self.storage.delete_model(model_id).await.map_err(|e| anyhow!("Failed to delete model: {}", e))?;

        {
            let mut models = self.models.write().await;
            models.remove(model_id);
        }

        info!("Deleted model: {}", model_id);
        Ok(())
    }

    pub async fn list_checkpoints(&self, model_id: &str) -> Result<Vec<u32>> {
        self.storage.list_checkpoints(model_id).await.map_err(|e| anyhow!("Failed to list checkpoints: {}", e))
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub async fn model_count(&self) -> usize {
        let models = self.models.read().await;
        models.len()
    }

    pub async fn cleanup_unused_models(&self, max_age_seconds: u64) -> Result<usize> {
        let now = chrono::Utc::now().timestamp() as u64;
        let mut removed = 0;

        {
            let mut models = self.models.write().await;
            let models_to_remove: Vec<String> =
                models.iter().filter(|(_, info)| now - info.last_used > max_age_seconds).map(|(id, _)| id.clone()).collect();

            for id in models_to_remove {
                models.remove(&id);
                removed += 1;
                info!("Cleaned up unused model: {}", id);
            }
        }

        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_manager() {
        let manager = ModelManager::new("/tmp/test_models_manager".to_string()).await.unwrap();

        let data = b"model weights".to_vec();
        let metrics = ModelMetrics { loss: 0.5, accuracy: Some(0.9), f1_score: None, precision: None, recall: None };

        let path = manager.store_model("test_model", &data, metrics.clone()).await.unwrap();
        assert!(path.contains("test_model"));

        let loaded = manager.load_model("test_model").await.unwrap();
        assert_eq!(loaded.id, "test_model");

        let count = manager.model_count().await;
        assert_eq!(count, 1);

        // Cleanup
        let _ = manager.delete_model("test_model").await;
    }

    #[tokio::test]
    async fn test_checkpoint_storage() {
        let manager = ModelManager::new("/tmp/test_models_checkpoint".to_string()).await.unwrap();

        let data = b"checkpoint data".to_vec();
        let metrics = ModelMetrics::default();

        let path = manager.store_checkpoint("test_model", 1, &data, metrics).await.unwrap();
        assert!(path.contains("checkpoint_1"));

        let loaded = manager.load_checkpoint("test_model", 1).await.unwrap();
        assert_eq!(data, loaded);

        let checkpoints = manager.list_checkpoints("test_model").await.unwrap();
        assert_eq!(checkpoints, vec![1]);

        // Cleanup
        let _ = manager.delete_model("test_model").await;
    }

    #[tokio::test]
    async fn test_get_model_checkpoint() {
        let manager = ModelManager::new("/tmp/test_models_get_checkpoint".to_string()).await.unwrap();

        let files = RawModelFiles {
            config: b"config".to_vec(),
            tokenizer: b"tokenizer".to_vec(),
            weights: b"weights".to_vec(),
        };
        manager.store_model_files("test_model", &files, ModelMetrics::default()).await.unwrap();

        let (checkpoint, loaded) = manager.get_model_checkpoint("test_model").await.unwrap();
        assert_eq!(loaded.config, files.config);
        assert_eq!(loaded.tokenizer, files.tokenizer);
        assert_eq!(loaded.weights, files.weights);
        assert!(checkpoint.verify_integrity(&files.weights));

        // Cleanup
        let _ = manager.delete_model("test_model").await;
    }
}
