use anyhow::{anyhow, bail, Context, Result};
use borsh::BorshDeserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

use crate::consensus::fedavg::{FedAvgAggregator, FedAvgConfig, WeightingStrategy};
use crate::rpc::messages::{GradientLayer, GradientPayload, GradientUpdate, TrainingBatch};

/// Decompress a possibly sparse `GradientLayer` into a full flattened `Vec<f32>`.
/// Dense layers are validated and cloned; compressed layers scatter the stored
/// values back into a zero vector of the original shape.
fn decompress_gradient_layer(layer: &GradientLayer) -> Result<Vec<f32>> {
    let total_len: usize = layer.shape.iter().product();
    if layer.indices.is_empty() {
        if layer.values.len() != total_len {
            bail!("Dense gradient size {} does not match shape product {}", layer.values.len(), total_len);
        }
        return Ok(layer.values.clone());
    }

    if layer.values.len() != layer.indices.len() {
        bail!("Compressed gradient has {} values but {} indices", layer.values.len(), layer.indices.len());
    }

    let mut full = vec![0.0f32; total_len];
    for (idx, value) in layer.indices.iter().zip(layer.values.iter()) {
        if *idx >= total_len {
            bail!("Gradient index {} out of bounds for shape {:?}", idx, layer.shape);
        }
        full[*idx] = *value;
    }
    Ok(full)
}

use super::checkpoint::{ModelCheckpoint, ModelMetrics};
use super::downloader::{download_model, is_valid_weights};
use super::storage::ModelStorage;
use super::{EncryptedModelFiles, RawModelFiles};

use candle_core::{DType, Device, Tensor};
use model_crypto;
use xenom_miner::model::DnaBert2Config;
use xenom_miner::tokenizer::DnaTokenizer;
use xenom_miner::trainer::DnaBert2Trainer;

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
    aggregators: Arc<RwLock<HashMap<String, FedAvgAggregator>>>,
    /// Cached trainable DNABERT-2 replicas used by the FedAvg aggregator.
    /// Each replica keeps its AdamW optimizer state across aggregation rounds.
    trainers: Arc<RwLock<HashMap<String, Arc<std::sync::Mutex<DnaBert2Trainer>>>>>,
    fedavg_config: FedAvgConfig,
    node_id: String,
}

impl ModelManager {
    pub async fn new(base_path: String) -> Result<Self> {
        let key = ModelStorage::generate_key();
        Self::new_with_key(base_path, key).await
    }

    pub async fn new_with_key(base_path: String, encryption_key: [u8; 32]) -> Result<Self> {
        let storage = Arc::new(ModelStorage::new(base_path.clone(), encryption_key));

        // Create base directory if it doesn't exist
        tokio::fs::create_dir_all(&base_path).await?;

        let node_id = Uuid::new_v4().to_string();

        let min_participants = std::env::var("FEDAVG_MIN_PARTICIPANTS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let fedavg_config = FedAvgConfig { min_participants, max_participants: 10, weighting_strategy: WeightingStrategy::Uniform };

        Ok(Self {
            base_path,
            storage,
            models: Arc::new(RwLock::new(HashMap::new())),
            aggregators: Arc::new(RwLock::new(HashMap::new())),
            trainers: Arc::new(RwLock::new(HashMap::new())),
            fedavg_config,
            node_id,
        })
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
    /// If the stored files cannot be decrypted (e.g. the encryption key changed) or the
    /// weights are not a valid checkpoint (e.g. a stale git-lfs pointer), they are removed
    /// and re-downloaded.
    pub async fn ensure_model_downloaded(&self, model_id: &str) -> Result<()> {
        // Check if a model file or checkpoint already exists for this id.
        if self.storage.model_exists(model_id).await {
            // Verify the files are actually loadable with the current key and contain valid weights.
            match self.storage.load_model_files(model_id).await {
                Ok(files) if is_valid_weights(&files.weights) => {
                    info!("Model {} already exists locally; skipping download", model_id);
                    return Ok(());
                }
                Ok(_) => {
                    info!(
                        "Model {} exists locally but weights look invalid (e.g. LFS pointer); removing and re-downloading",
                        model_id
                    );
                }
                Err(_) => {
                    info!("Model {} exists locally but cannot be decrypted; removing and re-downloading", model_id);
                }
            }
            self.delete_model(model_id).await?;
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
    pub async fn get_training_batch_with_genome(&self, model_id: &str, genome_root: Option<[u8; 32]>) -> Result<TrainingBatch> {
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

    /// Return the model checkpoint with still-encrypted files so they can be sent
    /// over the network and decrypted by the receiver.
    pub async fn get_encrypted_model_checkpoint(&self, model_id: &str) -> Result<(ModelCheckpoint, EncryptedModelFiles)> {
        // Load the model once to get the canonical plaintext weights hash and metadata.
        let model_info = self.load_model(model_id).await?;
        let files = self
            .storage
            .load_encrypted_model_files(model_id)
            .await
            .map_err(|e| anyhow!("Failed to load encrypted model files: {}", e))?;
        Ok((model_info.checkpoint, files))
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

    /// Submit an encrypted gradient update for FedAvg aggregation.
    ///
    /// If the aggregator reaches `min_participants` for all layers, the averaged
    /// gradient is applied to the model, a new checkpoint is stored, and its
    /// weights hash is returned.
    pub async fn submit_gradients(&self, update: &GradientUpdate) -> Result<Option<[u8; 32]>> {
        if update.participant_weight <= 0.0 {
            bail!("participant_weight must be positive");
        }

        let plaintext = model_crypto::decrypt(&update.encrypted_payload, self.storage.encryption_key())
            .context("Failed to decrypt gradient payload")?;
        let payload: GradientPayload =
            GradientPayload::try_from_slice(&plaintext).context("Failed to deserialize gradient payload")?;

        if payload.layer_gradients.is_empty() {
            bail!("Gradient payload contains no layers");
        }

        // Reject stale gradients that were computed on a checkpoint that is no longer
        // active. This prevents FedAvg from mixing gradients from different model
        // versions if a new checkpoint was produced between batch request and submission.
        let active_checkpoint = match self.get_model(&update.model_id).await {
            Some(info) => info.checkpoint.weights_hash,
            None => {
                self.load_model(&update.model_id)
                    .await
                    .with_context(|| format!("Model {} is not loaded and could not be loaded from storage", update.model_id))?
                    .checkpoint
                    .weights_hash
            }
        };

        if update.base_checkpoint != active_checkpoint {
            bail!(
                "Gradient for {} has base_checkpoint {} but active checkpoint is {}; rejecting stale update",
                update.model_id,
                hex::encode(update.base_checkpoint),
                hex::encode(active_checkpoint)
            );
        }

        let mut aggregators = self.aggregators.write().await;
        let aggregator =
            aggregators.entry(update.model_id.clone()).or_insert_with(|| FedAvgAggregator::new(self.fedavg_config.clone()));

        for (name, layer) in &payload.layer_gradients {
            let gradient =
                decompress_gradient_layer(layer).with_context(|| format!("Failed to decompress gradient for layer {}", name))?;
            aggregator
                .add_gradient(name, gradient, layer.shape.clone(), update.participant_weight)
                .with_context(|| format!("Failed to add gradient for layer {}", name))?;
        }

        if !aggregator.all_ready() {
            let first = payload.layer_gradients.keys().next().unwrap();
            info!(
                "Collected gradients for {} ({} of {} participants)",
                update.model_id,
                aggregator.participant_count(first),
                self.fedavg_config.min_participants
            );
            return Ok(None);
        }

        let averages = aggregator.compute_all_averages().context("Failed to compute averaged gradients")?;
        aggregator.reset();
        drop(aggregators);

        self.apply_averaged_gradients(&update.model_id, averages).await
    }

    /// Return a cached trainable DNABERT-2 replica for `model_id`, creating it from
    /// the stored checkpoint if necessary. Replicas are kept in memory so their
    /// AdamW optimizer state survives across FedAvg rounds.
    async fn get_or_create_trainer(&self, model_id: &str) -> Result<Arc<std::sync::Mutex<DnaBert2Trainer>>> {
        {
            let trainers = self.trainers.read().await;
            if let Some(trainer) = trainers.get(model_id) {
                return Ok(trainer.clone());
            }
        }

        let files = self.storage.load_model_files(model_id).await.map_err(|e| anyhow!("Failed to load model files: {}", e))?;
        let config = DnaBert2Config::from_bytes(&files.config).context("Failed to parse model config")?;
        let tokenizer = DnaTokenizer::from_bytes(&files.tokenizer).context("Failed to parse tokenizer")?;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());

        let trainer = DnaBert2Trainer::new(config, files.weights, tokenizer, Device::Cpu, threads, DType::F32)
            .context("Failed to load trainable model for aggregation")?;

        let mut trainers = self.trainers.write().await;
        let trainer = trainers.entry(model_id.to_string()).or_insert_with(|| Arc::new(std::sync::Mutex::new(trainer)));
        Ok(trainer.clone())
    }

    async fn apply_averaged_gradients(&self, model_id: &str, averages: HashMap<String, Vec<f32>>) -> Result<Option<[u8; 32]>> {
        info!("Aggregating gradients for {} and producing a new checkpoint", model_id);

        let trainer = self.get_or_create_trainer(model_id).await?;
        let base_path = self.base_path.clone();
        let model_id_owned = model_id.to_string();

        let (weights, new_hash_bytes) = tokio::task::spawn_blocking(move || {
            let trainer = trainer.lock().map_err(|e| anyhow!("Trainer mutex poisoned: {}", e))?;

            let data = trainer.varmap().data().lock().map_err(|e| anyhow!("VarMap poisoned: {}", e))?;
            let mut named_grads: HashMap<String, Tensor> = HashMap::with_capacity(averages.len());
            for (name, avg) in averages {
                let var = data.get(&name).ok_or_else(|| anyhow!("Model has no variable named {}", name))?;
                let shape = var.as_tensor().shape().clone();
                let grad = Tensor::from_vec(avg, shape, &Device::Cpu)
                    .with_context(|| format!("Failed to build gradient tensor for {}", name))?
                    .to_dtype(DType::F32)?;
                named_grads.insert(name, grad);
            }
            drop(data);

            // Use AdamW on the server so moment estimates persist across aggregation rounds.
            trainer.apply_gradients(&named_grads, 1e-5f32).context("Failed to apply averaged gradients")?;

            let tmp_name = format!("{}_fedavg_{}.safetensors", model_id_owned.replace('/', "_"), Uuid::new_v4());
            let tmp_path = std::path::Path::new(&base_path).join(&tmp_name);
            trainer.save_weights_to_path(&tmp_path).context("Failed to save updated weights")?;
            let weights = std::fs::read(&tmp_path).context("Failed to read updated weights")?;
            let _ = std::fs::remove_file(&tmp_path);

            let new_hash = blake3::hash(&weights);
            Ok::<_, anyhow::Error>((weights, <[u8; 32]>::from(new_hash)))
        })
        .await
        .context("Gradient aggregation task panicked")??;

        let files = self.storage.load_model_files(model_id).await.map_err(|e| anyhow!("Failed to load model files: {}", e))?;
        let new_files = RawModelFiles { config: files.config, tokenizer: files.tokenizer, weights };
        self.store_model_files(model_id, &new_files, ModelMetrics::default()).await?;

        info!("FedAvg produced new checkpoint for {}: hash {}", model_id, hex::encode(new_hash_bytes));

        Ok(Some(new_hash_bytes))
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

        let files = RawModelFiles { config: b"config".to_vec(), tokenizer: b"tokenizer".to_vec(), weights: b"weights".to_vec() };
        manager.store_model_files("test_model", &files, ModelMetrics::default()).await.unwrap();

        let (checkpoint, loaded) = manager.get_model_checkpoint("test_model").await.unwrap();
        assert_eq!(loaded.config, files.config);
        assert_eq!(loaded.tokenizer, files.tokenizer);
        assert_eq!(loaded.weights, files.weights);
        assert!(checkpoint.verify_integrity(&files.weights));

        // Cleanup
        let _ = manager.delete_model("test_model").await;
    }

    #[test]
    fn test_decompress_gradient_layer_dense_and_sparse() {
        let dense = GradientLayer { values: vec![1.0, 2.0, 3.0, 4.0], shape: vec![2, 2], indices: vec![] };
        assert_eq!(decompress_gradient_layer(&dense).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let sparse = GradientLayer { values: vec![5.0, 7.0], shape: vec![4], indices: vec![0, 3] };
        assert_eq!(decompress_gradient_layer(&sparse).unwrap(), vec![5.0, 0.0, 0.0, 7.0]);
    }
}
