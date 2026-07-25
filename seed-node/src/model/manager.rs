use anyhow::{anyhow, bail, Context, Result};
use borsh::BorshDeserialize;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::info;
use uuid::Uuid;

use crate::consensus::fedavg::{FedAvgAggregator, FedAvgConfig, WeightingStrategy};
use crate::rpc::messages::{GradientLayer, GradientPayload, GradientUpdate, TrainingBatch};

/// Decompress a possibly sparse `GradientLayer` into a full flattened `Vec<f32>`.
/// Dense layers are validated and cloned; compressed layers scatter the stored
/// values back into a zero vector of the original shape.
fn decompress_gradient_layer(layer: &GradientLayer) -> Result<Vec<f32>> {
    let total_len: usize = layer.shape.iter().product();

    // Reject gradients containing NaN/Inf before they can corrupt the model.
    for v in layer.values.iter() {
        if !v.is_finite() {
            bail!("Gradient value {} is not finite; rejecting update", v);
        }
    }

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
use xenom_miner::lora::LoraConfig;
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
    pub lora_config: Option<LoraConfig>,
}

/// A cached, trainable checkpoint lineage.
///
/// `base_hash` is the hash of the frozen base weights.
/// `head_hash` is the latest combined weights hash after the most recent aggregation.
/// `adapter_bytes` holds the serialized (plaintext) LoRA adapter when one exists.
/// The `trainer` holds the live DNABERT-2 weights and AdamW optimizer state,
/// and the `aggregator` collects gradients for the next FedAvg round.
pub struct CachedCheckpoint {
    pub base_hash: [u8; 32],
    pub head_hash: [u8; 32],
    pub trainer: DnaBert2Trainer,
    pub aggregator: FedAvgAggregator,
    pub last_used: Instant,
    pub model_id: String,
    /// Plaintext LoRA adapter bytes, kept in memory to serve adapter-only sync.
    pub adapter_bytes: Option<Vec<u8>>,
    /// When the current epoch started. `None` means no gradients have been
    /// collected yet for the next checkpoint.
    pub epoch_started: Option<Instant>,
    /// How long an epoch lasts before the collected gradients are aggregated.
    pub epoch_duration: Duration,
    /// Optional hard cap on the number of gradients per epoch (0 = disabled).
    pub epoch_size_cap: u32,
}

pub struct ModelManager {
    base_path: String,
    storage: Arc<ModelStorage>,
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    /// Maps a checkpoint hash (either an original base or a later head) to a
    /// shared, trainable checkpoint lineage. Multiple hashes may point to the
    /// same `CachedCheckpoint` so that stale gradients continue the same lineage.
    #[allow(clippy::type_complexity)]
    checkpoint_cache: Arc<RwLock<HashMap<[u8; 32], Arc<Mutex<CachedCheckpoint>>>>>,
    /// LRU ordering of the keys in `checkpoint_cache`, used for bounded eviction.
    cache_order: Arc<Mutex<VecDeque<[u8; 32]>>>,
    fedavg_config: FedAvgConfig,
    node_id: String,
    checkpoint_history_size: usize,
    lora_config: Option<LoraConfig>,
    /// Duration of a training epoch. Gradients submitted during an epoch are
    /// aggregated into a single new checkpoint when the epoch ends.
    epoch_duration: Duration,
    /// Optional hard cap on the number of gradients per epoch (0 = disabled).
    epoch_size_cap: u32,
    /// Maps a checkpoint hash to the hash of its direct parent, tracking the
    /// lineage of active checkpoints so stale-but-related bases can be rebased.
    lineage: Arc<RwLock<HashMap<[u8; 32], [u8; 32]>>>,
}

impl ModelManager {
    pub async fn new(base_path: String) -> Result<Self> {
        let key = ModelStorage::generate_key();
        Self::new_with_key(base_path, key, LoraConfig::from_env()).await
    }

    pub async fn new_with_key(base_path: String, encryption_key: [u8; 32], lora_config: Option<LoraConfig>) -> Result<Self> {
        let storage = Arc::new(ModelStorage::new(base_path.clone(), encryption_key));

        // Create base directory if it doesn't exist
        tokio::fs::create_dir_all(&base_path).await?;

        let node_id = Uuid::new_v4().to_string();

        // Epoch duration: how long the node collects gradients before producing a new
        // checkpoint. Defaults to 60s; set XENO_EPOCH_DURATION_SECS=0 to disable time-based
        // epochs and fall back to count-based aggregation.
        let epoch_duration_secs: u64 = std::env::var("XENO_EPOCH_DURATION_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
        let epoch_duration = Duration::from_secs(epoch_duration_secs);

        // Optional hard cap on the number of gradients per epoch (0 = disabled).
        // Reads XENO_FEDAVG_EPOCH_SIZE first, then the legacy FEDAVG_MIN_PARTICIPANTS.
        let epoch_size_cap: u32 = std::env::var("XENO_FEDAVG_EPOCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .or_else(|| std::env::var("FEDAVG_MIN_PARTICIPANTS").ok().and_then(|s| s.parse().ok()))
            .unwrap_or(0);

        // The FedAvg aggregator only needs a minimum of 1 participant to compute an
        // average; the epoch window / cap controls when aggregation is triggered.
        let fedavg_config = FedAvgConfig {
            min_participants: 1,
            max_participants: epoch_size_cap.max(10).max(1),
            weighting_strategy: WeightingStrategy::Uniform,
        };

        let checkpoint_history_size = std::env::var("XENO_CHECKPOINT_HISTORY_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(8);

        Ok(Self {
            base_path,
            storage,
            models: Arc::new(RwLock::new(HashMap::new())),
            checkpoint_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_order: Arc::new(Mutex::new(VecDeque::new())),
            fedavg_config,
            node_id,
            checkpoint_history_size,
            lora_config,
            epoch_duration,
            epoch_size_cap,
            lineage: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Return the LoRA configuration used by this manager.
    pub fn lora_config(&self) -> Option<&LoraConfig> {
        self.lora_config.as_ref()
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
        let (data, files_opt) = match self.storage.load_model_files(model_id).await {
            Ok(files) => {
                let data = files.weights.clone();
                (data, Some(files))
            }
            Err(_) => {
                let data = self.storage.load_model(model_id).await.map_err(|e| anyhow!("Failed to load model: {}", e))?;
                (data, None)
            }
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
            lora_config: self.lora_config.clone(),
        };

        // Cache the model
        {
            let mut models = self.models.write().await;
            models.insert(model_id.to_string(), model_info.clone());
        }

        // Initialise the checkpoint cache for the loaded model's checkpoint, if we have the files.
        if let Some(files) = files_opt {
            if let Ok(entry) = self.build_cached_checkpoint(model_id, model_info.checkpoint.weights_hash, files) {
                let mut cache = self.checkpoint_cache.write().await;
                let mut order = self.cache_order.lock().await;
                cache.entry(model_info.checkpoint.weights_hash).or_insert(entry);
                if !order.contains(&model_info.checkpoint.weights_hash) {
                    order.push_back(model_info.checkpoint.weights_hash);
                }
            }
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
                Ok(files) => {
                    let first_bytes: String = files.weights.iter().take(32).map(|b| format!("{:02x}", b)).collect();
                    info!(
                        "Model {} exists locally but weights do not look like a valid safetensors file ({} bytes, first bytes: {}); removing and re-downloading",
                        model_id,
                        files.weights.len(),
                        first_bytes
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
            lora_config: self.lora_config.clone(),
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
            lora_config: self.lora_config.clone(),
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

    /// Return checkpoint metadata for the V2 RPC, exposing both the combined
    /// checkpoint id and the base weights hash.
    pub async fn get_model_checkpoint_info_v2(&self, model_id: &str) -> Result<([u8; 32], [u8; 32])> {
        self.ensure_checkpoint_cached(model_id).await?;

        let active_hash = {
            let models = self.models.read().await;
            models.get(model_id).map(|info| info.checkpoint.weights_hash)
        };
        let active_hash = active_hash.ok_or_else(|| anyhow!("Model {} is not loaded", model_id))?;

        let cache = self.checkpoint_cache.read().await;
        let entry = cache.get(&active_hash).ok_or_else(|| anyhow!("Active checkpoint not cached"))?;
        let entry = entry.lock().await;
        Ok((entry.head_hash, entry.base_hash))
    }

    /// Return encrypted model files for the V2 RPC.
    ///
    /// If `cached_base_hash` matches the active base hash and a LoRA adapter is
    /// available, only the encrypted adapter is returned (`is_adapter=true`).
    /// Otherwise the full merged checkpoint is returned (`is_adapter=false`).
    pub async fn get_encrypted_model_checkpoint_v2(
        &self,
        model_id: &str,
        cached_base_hash: Option<[u8; 32]>,
    ) -> Result<(EncryptedModelFiles, [u8; 32], [u8; 32], bool)> {
        self.ensure_checkpoint_cached(model_id).await?;

        let active_hash = {
            let models = self.models.read().await;
            models.get(model_id).map(|info| info.checkpoint.weights_hash)
        };
        let active_hash = active_hash.ok_or_else(|| anyhow!("Model {} is not loaded", model_id))?;

        let cache = self.checkpoint_cache.read().await;
        let entry = cache.get(&active_hash).ok_or_else(|| anyhow!("Active checkpoint not cached"))?.clone();
        let entry = entry.lock().await;

        let metadata = self
            .storage
            .load_encrypted_model_metadata(model_id)
            .await
            .map_err(|e| anyhow!("Failed to load encrypted model metadata: {}", e))?;

        let base_hash = entry.base_hash;
        let head_hash = entry.head_hash;

        let send_adapter = cached_base_hash == Some(base_hash) && entry.adapter_bytes.is_some();
        let weights = if send_adapter {
            let adapter = entry.adapter_bytes.as_ref().unwrap();
            model_crypto::encrypt(adapter, self.storage.encryption_key()).context("Failed to encrypt adapter")?
        } else {
            self.storage
                .load_encrypted_model_files(model_id)
                .await
                .map_err(|e| anyhow!("Failed to load encrypted model files: {}", e))?
                .weights
        };

        Ok((
            EncryptedModelFiles { config: metadata.config, tokenizer: metadata.tokenizer, weights },
            head_hash,
            base_hash,
            send_adapter,
        ))
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

    /// Check whether `ancestor` is in the active checkpoint lineage for `model_id`.
    /// This allows stale-but-related bases to be accepted for block rewards and
    /// rebased onto the current active checkpoint.
    pub async fn is_ancestor_of_active(&self, model_id: &str, ancestor: [u8; 32]) -> bool {
        let active = match self.get_model(model_id).await {
            Some(info) => info.checkpoint.weights_hash,
            None => return false,
        };
        if ancestor == active {
            return true;
        }
        let lineage = self.lineage.read().await;
        let mut current = active;
        for _ in 0..self.checkpoint_history_size.saturating_add(1) {
            match lineage.get(&current) {
                Some(parent) if *parent == ancestor => return true,
                Some(parent) => current = *parent,
                None => break,
            }
        }
        false
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
    /// The update is accepted if its `base_checkpoint` is present in the bounded
    /// checkpoint cache. The averaged gradient is applied to the matching
    /// checkpoint lineage. A new checkpoint hash is returned only when the
    /// updated lineage was the active model checkpoint.
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

        // Make sure the target model and its active checkpoint are cached.
        self.ensure_checkpoint_cached(&update.model_id).await?;

        // Look up the base checkpoint in the bounded cache. If it is absent, the
        // gradient is too stale to be accepted.
        let entry = {
            let cache = self.checkpoint_cache.read().await;
            match cache.get(&update.base_checkpoint).cloned() {
                Some(entry) => entry,
                None => {
                    bail!(
                        "Gradient for {} has base_checkpoint {} but it is not in the checkpoint cache; rejecting stale update",
                        update.model_id,
                        hex::encode(update.base_checkpoint)
                    );
                }
            }
        };

        // Update LRU ordering for the base key.
        self.touch_cache_key(update.base_checkpoint).await;

        let (old_head_hash, averages) = {
            let mut entry_guard = entry.lock().await;
            entry_guard.last_used = Instant::now();

            if entry_guard.model_id != update.model_id {
                bail!("Gradient base checkpoint belongs to model {} not {}", entry_guard.model_id, update.model_id);
            }

            for (name, layer) in &payload.layer_gradients {
                let gradient =
                    decompress_gradient_layer(layer).with_context(|| format!("Failed to decompress gradient for layer {}", name))?;
                entry_guard
                    .aggregator
                    .add_gradient(name, gradient, layer.shape.clone(), update.participant_weight)
                    .with_context(|| format!("Failed to add gradient for layer {}", name))?;
            }

            // Start the epoch timer on the first gradient of a new epoch.
            let first_layer = payload.layer_gradients.keys().next().unwrap();
            if entry_guard.epoch_started.is_none() {
                entry_guard.epoch_started = Some(Instant::now());
            }
            let participant_count = entry_guard.aggregator.participant_count(first_layer);

            // Decide whether the epoch has ended: either the time window expired
            // or the optional hard cap on gradients was reached.
            let elapsed = entry_guard.epoch_started.unwrap().elapsed();
            let ready_by_time = elapsed >= entry_guard.epoch_duration;
            let ready_by_count = entry_guard.epoch_size_cap > 0 && participant_count >= entry_guard.epoch_size_cap;

            if !ready_by_time && !ready_by_count {
                info!(
                    "Collected gradients for {} ({} participants, {}/{}s elapsed)",
                    update.model_id,
                    participant_count,
                    elapsed.as_secs(),
                    entry_guard.epoch_duration.as_secs()
                );
                return Ok(None);
            }

            let averages = entry_guard.aggregator.compute_all_averages().context("Failed to compute averaged gradients")?;
            let old_head = entry_guard.head_hash;
            entry_guard.aggregator.reset();
            entry_guard.epoch_started = None;
            (old_head, averages)
        };

        // Apply the averaged gradients on a blocking thread so the async runtime
        // is not paused by the DNABERT-2 forward / backward pass.
        let entry_clone = entry.clone();
        let is_lora = self.lora_config.is_some();
        let (weights, new_hash) = tokio::task::spawn_blocking(move || {
            let mut entry = entry_clone.blocking_lock();

            let data = entry.trainer.varmap().data().lock().map_err(|e| anyhow!("VarMap poisoned: {}", e))?;
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
            entry.trainer.apply_gradients(&named_grads, 1e-5f32).context("Failed to apply averaged gradients")?;

            // Serialize weights in memory to avoid a temporary disk round-trip.
            let weights = entry.trainer.save_weights_to_bytes().context("Failed to serialize updated weights")?;
            let new_hash = blake3::hash(&weights);
            let new_hash_bytes = <[u8; 32]>::from(new_hash);

            entry.head_hash = new_hash_bytes;
            entry.last_used = Instant::now();

            // Refresh the in-memory adapter bytes for Phase 2 adapter-only sync.
            if is_lora {
                entry.adapter_bytes = entry.trainer.save_adapter_to_bytes().ok();
            }

            Ok::<_, anyhow::Error>((weights, new_hash_bytes))
        })
        .await
        .context("Gradient aggregation task panicked")??;

        // Determine whether this lineage was the active checkpoint before the update.
        let old_active = self.get_model(&update.model_id).await.map(|info| info.checkpoint.weights_hash);
        let promoted_to_active = old_active == Some(old_head_hash);

        if promoted_to_active {
            let files = self
                .storage
                .load_model_metadata(&update.model_id)
                .await
                .map_err(|e| anyhow!("Failed to load model metadata: {}", e))?;
            let new_files = RawModelFiles { config: files.config, tokenizer: files.tokenizer, weights };
            self.store_model_files(&update.model_id, &new_files, ModelMetrics::default()).await?;
            info!("FedAvg produced new active checkpoint for {}: hash {}", update.model_id, hex::encode(new_hash));
        } else {
            info!("FedAvg produced checkpoint for {}: hash {} (not active)", update.model_id, hex::encode(new_hash));
        }

        // Track the lineage so we can rebase or accept stale-but-related blocks later.
        {
            let mut lineage = self.lineage.write().await;
            lineage.insert(new_hash, old_head_hash);
        }

        // Insert a new cache entry keyed by the new head so that future gradients
        // for this checkpoint continue the same lineage.
        {
            let mut cache = self.checkpoint_cache.write().await;
            let mut order = self.cache_order.lock().await;
            cache.insert(new_hash, entry.clone());
            order.push_back(new_hash);
            let active_for_eviction = if promoted_to_active { new_hash } else { old_active.unwrap_or(new_hash) };
            self.evict_lru_not_active(&mut cache, &mut order, active_for_eviction);
        }

        if promoted_to_active {
            Ok(Some(new_hash))
        } else {
            Ok(None)
        }
    }

    /// Ensure that the active checkpoint for `model_id` is represented in the
    /// checkpoint cache, creating a trainable replica from storage if needed.
    async fn ensure_checkpoint_cached(&self, model_id: &str) -> Result<()> {
        {
            let models = self.models.read().await;
            if !models.contains_key(model_id) {
                drop(models);
                self.load_model(model_id).await?;
            }
        }

        let active_hash = {
            let models = self.models.read().await;
            models.get(model_id).map(|info| info.checkpoint.weights_hash)
        };

        if let Some(active_hash) = active_hash {
            let cache = self.checkpoint_cache.read().await;
            if !cache.contains_key(&active_hash) {
                drop(cache);
                let files = self.storage.load_model_files(model_id).await.map_err(|e| anyhow!("Failed to load model files: {}", e))?;
                let entry = self.build_cached_checkpoint(model_id, active_hash, files)?;
                let mut cache = self.checkpoint_cache.write().await;
                let mut order = self.cache_order.lock().await;
                cache.entry(active_hash).or_insert(entry);
                if !order.contains(&active_hash) {
                    order.push_back(active_hash);
                }
            }
        }

        Ok(())
    }

    /// Move `key` to the back of the LRU order list.
    async fn touch_cache_key(&self, key: [u8; 32]) {
        let mut order = self.cache_order.lock().await;
        if let Some(pos) = order.iter().position(|h| *h == key) {
            order.remove(pos);
        }
        order.push_back(key);
    }

    /// Evict the least-recently-used cache key that is not the active checkpoint
    /// until the cache size is within the configured history bound.
    fn evict_lru_not_active(
        &self,
        cache: &mut HashMap<[u8; 32], Arc<Mutex<CachedCheckpoint>>>,
        order: &mut VecDeque<[u8; 32]>,
        active: [u8; 32],
    ) {
        while order.len() > self.checkpoint_history_size {
            let mut victim = None;
            for (i, key) in order.iter().enumerate() {
                if *key != active {
                    victim = Some(i);
                    break;
                }
            }
            match victim {
                Some(i) => {
                    let key = order.remove(i).expect("index valid");
                    cache.remove(&key);
                }
                None => break,
            }
        }
    }

    /// Build a cached trainable DNABERT-2 replica from raw model files.
    fn build_cached_checkpoint(
        &self,
        model_id: &str,
        active_hash: [u8; 32],
        files: RawModelFiles,
    ) -> Result<Arc<Mutex<CachedCheckpoint>>> {
        let config = DnaBert2Config::from_bytes(&files.config).context("Failed to parse model config")?;
        let tokenizer = DnaTokenizer::from_bytes(&files.tokenizer).context("Failed to parse tokenizer")?;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());

        let trainer =
            DnaBert2Trainer::new(config, files.weights, tokenizer, Device::Cpu, threads, DType::F32, self.lora_config.clone())
                .context("Failed to load trainable model for aggregation")?;

        // Compute the base weights hash and, for LoRA, the adapter bytes.
        let base_bytes = trainer.save_base_weights_to_bytes().context("Failed to serialize base weights")?;
        let base_hash = <[u8; 32]>::from(blake3::hash(&base_bytes));
        let adapter_bytes = if self.lora_config.is_some() {
            Some(trainer.save_adapter_to_bytes().context("Failed to serialize adapter weights")?)
        } else {
            None
        };

        let entry = CachedCheckpoint {
            base_hash,
            head_hash: active_hash,
            trainer,
            aggregator: FedAvgAggregator::new(self.fedavg_config.clone()),
            last_used: Instant::now(),
            model_id: model_id.to_string(),
            adapter_bytes,
            epoch_started: None,
            epoch_duration: self.epoch_duration,
            epoch_size_cap: self.epoch_size_cap,
        };

        Ok(Arc::new(Mutex::new(entry)))
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
