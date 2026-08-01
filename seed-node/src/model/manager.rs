use anyhow::{anyhow, bail, Context, Result};
use borsh::BorshDeserialize;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

use crate::consensus::fedavg::{FedAvgAggregator, FedAvgConfig, WeightingStrategy};
use crate::rpc::messages::{GradientLayer, GradientPayload, GradientUpdate, TrainingBatch};

/// Build a `HashMap` of named gradient tensors from averaged gradient vectors,
/// using the trainer's variable shapes and casting to F32 on CPU.
fn build_named_grads(trainer: &DnaBert2Trainer, averages: HashMap<String, Vec<f32>>) -> Result<HashMap<String, Tensor>> {
    let data = trainer.varmap().data().lock().map_err(|e| anyhow!("VarMap poisoned: {}", e))?;
    let mut named_grads = HashMap::with_capacity(averages.len());
    for (name, avg) in averages {
        let var = data.get(&name).ok_or_else(|| anyhow!("Model has no variable named {}", name))?;
        let shape = var.as_tensor().shape().clone();
        let grad =
            Tensor::from_vec(avg, shape, &Device::Cpu).with_context(|| format!("Failed to build gradient tensor for {}", name))?;
        named_grads.insert(name, grad.to_dtype(DType::F32)?);
    }
    Ok(named_grads)
}

/// Compute a deterministic commitment hash over named gradient tensors.
/// Mirrors `xenom_miner::trainer::gradient::gradient_commitment`.
fn gradient_commitment(grads: &HashMap<String, Tensor>) -> Result<[u8; 32]> {
    let mut hasher = blake3::Hasher::new();
    let mut names: Vec<_> = grads.keys().cloned().collect();
    names.sort();
    for name in names {
        let grad = &grads[&name];
        let grad_f32 = grad.to_dtype(DType::F32)?;
        let values = grad_f32.flatten_all()?.to_vec1::<f32>()?;
        hasher.update(name.as_bytes());
        for value in values {
            hasher.update(&value.to_le_bytes());
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

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

/// Load a safetensors buffer into a CPU F32 weight snapshot.
fn load_weights_snapshot(weights_bytes: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    candle_core::safetensors::load_buffer(weights_bytes, device).context("Failed to load safetensors weights into snapshot")
}

/// Serialize a weight snapshot to a safetensors byte vector.
fn serialize_weight_snapshot(weights: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    let tensors: Vec<(String, &Tensor)> = weights.iter().map(|(k, v)| (k.clone(), v)).collect();
    safetensors::tensor::serialize(tensors, &None).map_err(|e| anyhow!("Failed to serialize MGM-1 weights: {}", e))
}

/// Add an averaged weight-space delta to a weight snapshot.
/// The `averages` are now weight deltas produced by the miner's local AdamW
/// step, so `lr` is kept at 1.0 and the delta is added (not subtracted).
fn apply_weight_delta_to_snapshot(
    weights: &HashMap<String, Tensor>,
    averages: &HashMap<String, Vec<f32>>,
    lr: f64,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut updated = HashMap::with_capacity(weights.len());
    for (name, theta) in weights {
        let delta_vec = match averages.get(name) {
            Some(d) => d,
            None => continue,
        };
        let shape = theta.shape().clone();
        let delta = Tensor::from_vec(delta_vec.clone(), shape, device)
            .with_context(|| format!("Failed to build delta tensor for {}", name))?
            .to_dtype(theta.dtype())?;
        let delta = delta.to_device(theta.device())?;
        let delta_scaled = (delta * lr)?;
        let next = (theta + &delta_scaled)?;
        updated.insert(name.clone(), next);
    }
    Ok(updated)
}

/// Compute `updated - base` for each shared weight.
fn compute_weight_delta(base: &HashMap<String, Tensor>, updated: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut delta = HashMap::with_capacity(base.len());
    for (name, base_t) in base {
        let updated_t = updated.get(name).ok_or_else(|| anyhow!("Updated weights missing variable {}", name))?;
        let d = (updated_t - base_t)?;
        delta.insert(name.clone(), d);
    }
    Ok(delta)
}

/// Add a weight-space delta to an active checkpoint snapshot.
fn add_weight_delta(base: &HashMap<String, Tensor>, delta: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut result = HashMap::with_capacity(base.len());
    for (name, base_t) in base {
        let new = if let Some(d) = delta.get(name) { (base_t + d)? } else { base_t.copy().context("Failed to copy tensor")? };
        result.insert(name.clone(), new);
    }
    Ok(result)
}

/// Build the new MGM-1 checkpoint from `files` and averaged weight deltas.
///
/// `averages` now contain the weight-space change (`new - old`) produced by the
/// miner's local AdamW step, not raw gradients. If `base_snapshot` is `Some`,
/// the delta is first added to that stale base, the resulting weight-space
/// delta is computed, and it is added to the active weights from `files`. For
/// an active-base update `base_snapshot` is `None` and the delta is added
/// directly to the active weights.
///
/// Returns the serialized new weights, their hash, and the old active weights
/// snapshot so the caller can store it for future rebases.
#[allow(clippy::type_complexity)]
fn apply_mgm1_gradients_sync(
    files: RawModelFiles,
    averages: HashMap<String, Vec<f32>>,
    base_snapshot: Option<HashMap<String, Tensor>>,
) -> Result<(Vec<u8>, [u8; 32], HashMap<String, Tensor>)> {
    let device = Device::Cpu;
    let active_weights = load_weights_snapshot(&files.weights, &device)?;

    let new_weights = if let Some(base) = base_snapshot {
        let rebased =
            apply_weight_delta_to_snapshot(&base, &averages, 1.0, &device).context("Failed to apply delta to stale MGM-1 base")?;
        let delta = compute_weight_delta(&base, &rebased).context("Failed to compute MGM-1 rebase delta")?;
        add_weight_delta(&active_weights, &delta).context("Failed to apply MGM-1 rebase delta")?
    } else {
        apply_weight_delta_to_snapshot(&active_weights, &averages, 1.0, &device)
            .context("Failed to apply delta to active MGM-1 weights")?
    };

    let weights = serialize_weight_snapshot(&new_weights).context("Failed to serialize updated MGM-1 weights")?;
    let new_hash = <[u8; 32]>::from(blake3::hash(&weights));
    Ok((weights, new_hash, active_weights))
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
use xenom_miner::trainer::{DnaBert2Trainer, ManualAdamW};

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

/// A snapshot of a checkpoint's trainable weights and optimizer state.
/// Kept per-head so that stale-but-related gradients can be rebased onto the
/// active checkpoint using the exact base they were computed against.
#[derive(Clone)]
pub struct CheckpointSnapshot {
    pub weights: HashMap<String, Tensor>,
    pub optimizer: ManualAdamW,
}

/// A pending rebase for a stale base that is still in the active lineage.
/// Gradients for this base are aggregated separately and flushed onto the
/// active checkpoint when the rebase epoch expires or its cap is reached.
pub struct PendingRebase {
    pub epoch_started: Option<Instant>,
    pub aggregator: FedAvgAggregator,
    pub epoch_duration: Duration,
    pub epoch_size_cap: u32,
}

impl PendingRebase {
    pub fn new(fedavg_config: FedAvgConfig, epoch_duration: Duration, epoch_size_cap: u32) -> Self {
        Self { epoch_started: None, aggregator: FedAvgAggregator::new(fedavg_config), epoch_duration, epoch_size_cap }
    }
}

/// A cached, trainable checkpoint lineage.
///
/// `base_hash` is the hash of the frozen base weights.
/// `head_hash` is the latest combined weights hash after the most recent aggregation.
/// `adapter_bytes` holds the serialized (plaintext) LoRA adapter when one exists.
/// The `trainer` holds the live DNABERT-2 weights and AdamW optimizer state when
/// the model is trainable on this node; `None` for models such as MGM-1 whose
/// FedAvg path is not yet implemented.
pub struct CachedCheckpoint {
    pub base_hash: [u8; 32],
    pub head_hash: [u8; 32],
    pub trainer: Option<DnaBert2Trainer>,
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
    /// Snapshots of previous heads in this lineage. The key is the head hash
    /// and the value holds the trainable weights and optimizer state at that point.
    pub snapshots: HashMap<[u8; 32], CheckpointSnapshot>,
    /// Gradients for stale-but-related bases that are waiting to be rebased.
    pub pending_rebases: HashMap<[u8; 32], PendingRebase>,
    /// FedAvg configuration shared by this checkpoint's aggregators.
    pub fedavg_config: FedAvgConfig,
}

pub struct ModelManager {
    base_path: String,
    storage: Arc<ModelStorage>,
    config_file: Option<PathBuf>,
    tokenizer_file: Option<PathBuf>,
    weights_file: Option<PathBuf>,
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
    /// Maps a training block number to the active checkpoint hash at that block,
    /// enabling historical model evaluation and learning curves.
    checkpoint_history: Arc<RwLock<BTreeMap<u64, [u8; 32]>>>,
}

impl ModelManager {
    /// Return the base directory where model data is stored.
    pub fn base_path(&self) -> &str {
        &self.base_path
    }

    pub async fn new(base_path: String) -> Result<Self> {
        let key = ModelStorage::generate_key();
        Self::new_with_key(base_path, key, LoraConfig::from_env(), None, None, None).await
    }

    pub async fn new_with_key(
        base_path: String,
        encryption_key: [u8; 32],
        lora_config: Option<LoraConfig>,
        config_file: Option<PathBuf>,
        tokenizer_file: Option<PathBuf>,
        weights_file: Option<PathBuf>,
    ) -> Result<Self> {
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
            config_file,
            tokenizer_file,
            weights_file,
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
            checkpoint_history: Arc::new(RwLock::new(BTreeMap::new())),
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
            match self.build_cached_checkpoint(model_id, model_info.checkpoint.weights_hash, files) {
                Ok(entry) => {
                    let mut cache = self.checkpoint_cache.write().await;
                    let mut order = self.cache_order.lock().await;
                    cache.entry(model_info.checkpoint.weights_hash).or_insert(entry);
                    if !order.contains(&model_info.checkpoint.weights_hash) {
                        order.push_back(model_info.checkpoint.weights_hash);
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to build cached checkpoint for {} (hash {}): {:#}. Inference/training may fail until the model can be loaded.",
                        model_id,
                        hex::encode(model_info.checkpoint.weights_hash),
                        e
                    );
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
    pub async fn ensure_model_downloaded(&self, model_id: &str, from_scratch: bool) -> Result<()> {
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

        if from_scratch {
            info!("Model {} not found locally; downloading config/tokenizer for from-scratch training", model_id);
        } else {
            info!("Model {} not found locally; downloading from Hugging Face", model_id);
        }
        let files = download_model(
            model_id,
            from_scratch,
            self.config_file.as_deref(),
            self.tokenizer_file.as_deref(),
            self.weights_file.as_deref(),
        )
        .await?;

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

        // Preserve runtime flags (loaded / last_used) if the model is already known,
        // so a checkpoint update does not flip the model to "inactive" in gRPC info.
        let (loaded, last_used) = {
            let models = self.models.read().await;
            models.get(model_id).map(|info| (info.loaded, info.last_used)).unwrap_or((false, 0))
        };

        let model_info = ModelInfo {
            id: model_id.to_string(),
            name: model_id.to_string(),
            version: 1,
            category: "NLP".to_string(),
            checkpoint,
            loaded,
            last_used,
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

    /// Return the active checkpoint hash for `model_id`, if known.
    pub async fn active_hash(&self, model_id: &str) -> Option<[u8; 32]> {
        self.get_model(model_id).await.map(|info| info.checkpoint.weights_hash)
    }

    /// Record the active checkpoint hash for a given training block number.
    pub async fn record_checkpoint(&self, model_id: &str, block_number: u64) -> Result<()> {
        let hash = self
            .active_hash(model_id)
            .await
            .ok_or_else(|| anyhow!("No active checkpoint for model {} to record at block {}", model_id, block_number))?;
        let mut history = self.checkpoint_history.write().await;
        history.insert(block_number, hash);
        Ok(())
    }

    /// Load a historical checkpoint into the active slot so it can be used for
    /// block-height-specific inference. The current active checkpoint is preserved
    /// in storage and can be restored by loading the model without a block height.
    pub async fn load_historical_checkpoint(&self, model_id: &str, weights_hash: [u8; 32]) -> Result<()> {
        let files = self
            .storage
            .load_model_files_by_hash(model_id, weights_hash)
            .await
            .map_err(|e| anyhow!("Failed to load historical checkpoint for {}: {}", model_id, e))?;

        // Temporarily store the historical files as the active checkpoint.
        self.storage
            .store_model_files(model_id, &files)
            .await
            .map_err(|e| anyhow!("Failed to activate historical checkpoint: {}", e))?;

        // Reload the model metadata / weights from disk.
        self.load_model(model_id).await.map_err(|e| anyhow!("Failed to load model after activating historical checkpoint: {}", e))?;

        // Update the in-memory active hash so it matches the requested historical one.
        {
            let mut models = self.models.write().await;
            if let Some(model) = models.get_mut(model_id) {
                model.checkpoint.weights_hash = weights_hash;
            }
        }

        info!("Switched {} to historical checkpoint {}", model_id, hex::encode(weights_hash));
        Ok(())
    }

    /// Return the active checkpoint hash at `block_number`, if recorded.
    pub async fn checkpoint_at(&self, block_number: u64) -> Option<[u8; 32]> {
        let history = self.checkpoint_history.read().await;
        // Return the checkpoint active at or before the requested block.
        history.range(..=block_number).next_back().map(|(_, h)| *h)
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
    /// checkpoint cache or is an ancestor of the active checkpoint. When the base
    /// is the active checkpoint, the averaged gradient is applied directly. When
    /// the base is a stale but related checkpoint, the averaged gradient is first
    /// applied to the snapshot of that base to compute a delta; the delta is then
    /// added to the active checkpoint (delta rebase).
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

        // Reconstruct the decrypted payload as named tensors and verify that its
        // commitment matches the one signed in the update. This prevents a miner
        // from claiming one set of gradients while sending another.
        let mut reconstructed = HashMap::with_capacity(payload.layer_gradients.len());
        for (name, layer) in &payload.layer_gradients {
            let flat = decompress_gradient_layer(layer)?;
            let tensor = Tensor::from_vec(flat, layer.shape.clone(), &Device::Cpu)?;
            reconstructed.insert(name.clone(), tensor);
        }
        let expected = update.gradients_commitment;
        let actual = gradient_commitment(&reconstructed)?;
        if actual != expected {
            bail!(
                "Gradient commitment mismatch for {}: expected {}, got {}. Rejecting update.",
                update.model_id,
                hex::encode(expected),
                hex::encode(actual)
            );
        }

        // Make sure the target model and its active checkpoint are cached.
        self.ensure_checkpoint_cached(&update.model_id).await?;

        // Look up the base checkpoint in the bounded cache. If it is absent, the
        // gradient may still be valid if it is an ancestor of the active checkpoint.
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

        let active_hash =
            self.active_hash(&update.model_id).await.ok_or_else(|| anyhow!("No active checkpoint for {}", update.model_id))?;
        let base = update.base_checkpoint;
        let is_active = base == active_hash;

        if !is_active && !self.is_ancestor_of_active(&update.model_id, base).await {
            bail!(
                "Gradient base {} is not active nor an ancestor of active {} for {}",
                hex::encode(base),
                hex::encode(active_hash),
                update.model_id
            );
        }

        let first_layer = payload.layer_gradients.keys().next().unwrap().clone();
        let (old_head, averages, base_snapshot) = {
            let mut entry_guard = entry.lock().await;
            entry_guard.last_used = Instant::now();

            if entry_guard.model_id != update.model_id {
                bail!("Gradient base checkpoint belongs to model {} not {}", entry_guard.model_id, update.model_id);
            }

            let base_snapshot = if is_active {
                None
            } else {
                Some(
                    entry_guard
                        .snapshots
                        .get(&base)
                        .map(|s| s.weights.clone())
                        .ok_or_else(|| anyhow!("Missing snapshot for rebase base {}", hex::encode(base)))?,
                )
            };

            if is_active {
                // Active checkpoint: aggregate directly into the main aggregator.
                for (name, layer) in &payload.layer_gradients {
                    let gradient = decompress_gradient_layer(layer)
                        .with_context(|| format!("Failed to decompress gradient for layer {}", name))?;
                    entry_guard
                        .aggregator
                        .add_gradient(name, gradient, layer.shape.clone(), update.participant_weight)
                        .with_context(|| format!("Failed to add gradient for layer {}", name))?;
                }

                if entry_guard.epoch_started.is_none() {
                    entry_guard.epoch_started = Some(Instant::now());
                }
                let participant_count = entry_guard.aggregator.participant_count(&first_layer);
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
                (old_head, averages, base_snapshot)
            } else {
                // Stale-but-related base: aggregate into a separate pending rebase.
                let config = entry_guard.fedavg_config.clone();
                let duration = entry_guard.epoch_duration;
                let cap = entry_guard.epoch_size_cap;
                let pending = entry_guard.pending_rebases.entry(base).or_insert_with(|| PendingRebase::new(config, duration, cap));

                for (name, layer) in &payload.layer_gradients {
                    let gradient = decompress_gradient_layer(layer)
                        .with_context(|| format!("Failed to decompress gradient for layer {}", name))?;
                    pending
                        .aggregator
                        .add_gradient(name, gradient, layer.shape.clone(), update.participant_weight)
                        .with_context(|| format!("Failed to add gradient for layer {}", name))?;
                }

                if pending.epoch_started.is_none() {
                    pending.epoch_started = Some(Instant::now());
                }
                let participant_count = pending.aggregator.participant_count(&first_layer);
                let elapsed = pending.epoch_started.unwrap().elapsed();
                let ready_by_time = elapsed >= pending.epoch_duration;
                let ready_by_count = pending.epoch_size_cap > 0 && participant_count >= pending.epoch_size_cap;

                if !ready_by_time && !ready_by_count {
                    info!(
                        "Collected rebase gradients for {} base {} ({} participants, {}/{}s elapsed)",
                        update.model_id,
                        hex::encode(base),
                        participant_count,
                        elapsed.as_secs(),
                        pending.epoch_duration.as_secs()
                    );
                    return Ok(None);
                }

                let averages = pending.aggregator.compute_all_averages().context("Failed to compute averaged gradients")?;
                entry_guard.pending_rebases.remove(&base);
                (entry_guard.head_hash, averages, base_snapshot)
            }
        };

        // MGM-1: load the current model files, apply the averaged gradients on CPU,
        // and support both active and stale (delta-rebase) base checkpoints.
        if update.model_id.contains("mgm-1") {
            let files = self
                .storage
                .load_model_files(&update.model_id)
                .await
                .map_err(|e| anyhow!("Failed to load MGM-1 model files: {}", e))?;
            let (weights, new_hash, old_active_snapshot) =
                tokio::task::spawn_blocking(move || apply_mgm1_gradients_sync(files, averages, base_snapshot))
                    .await
                    .context("MGM-1 gradient aggregation task panicked")??;

            {
                let mut entry_guard = entry.lock().await;
                entry_guard.head_hash = new_hash;
                entry_guard.last_used = Instant::now();
                entry_guard
                    .snapshots
                    .insert(old_head, CheckpointSnapshot { weights: old_active_snapshot, optimizer: ManualAdamW::new(1e-5) });
            }

            self.finalize_new_head(&update.model_id, weights, old_head, new_hash, entry.clone()).await?;
            info!("FedAvg produced new active checkpoint for {}: hash {}", update.model_id, hex::encode(new_hash));
            return Ok(Some(new_hash));
        }

        // Apply the averaged gradients on a blocking thread so the async runtime
        // is not paused by the DNABERT-2 forward / backward pass.
        let entry_clone = entry.clone();
        let is_lora = self.lora_config.is_some();
        let rebase_base = if is_active { None } else { Some(base) };
        let (weights, new_hash) = tokio::task::spawn_blocking(move || {
            let mut entry = entry_clone.blocking_lock();

            if let Some(rebase_base) = rebase_base {
                // Delta rebase: apply the gradient to the stale base snapshot, then
                // add the resulting weight-space delta to the active checkpoint.
                let snapshot = entry
                    .snapshots
                    .get(&rebase_base)
                    .cloned()
                    .ok_or_else(|| anyhow!("Missing snapshot for rebase base {}", hex::encode(rebase_base)))?;

                let (weights, new_hash, old_active_snapshot, adapter_bytes) = {
                    let trainer = entry.trainer.as_ref().ok_or_else(|| anyhow!("Cached checkpoint has no trainer"))?;
                    let named_grads = build_named_grads(trainer, averages)?;

                    let old_active_snapshot = CheckpointSnapshot {
                        weights: trainer.trainable_weights().context("Failed to snapshot active weights")?,
                        optimizer: trainer.clone_optimizer().context("Failed to snapshot active optimizer")?,
                    };

                    let delta = trainer
                        .compute_delta_from_snapshot(&snapshot.weights, &snapshot.optimizer, &named_grads, 1e-5f32)
                        .context("Failed to compute rebase delta")?;
                    trainer.apply_weight_delta(&delta).context("Failed to apply rebase delta")?;

                    let weights = trainer.save_weights_to_bytes().context("Failed to serialize rebased weights")?;
                    let new_hash = <[u8; 32]>::from(blake3::hash(&weights));
                    let adapter_bytes = if is_lora { trainer.save_adapter_to_bytes().ok() } else { None };
                    (weights, new_hash, old_active_snapshot, adapter_bytes)
                };

                entry.snapshots.insert(old_head, old_active_snapshot);
                entry.head_hash = new_hash;
                entry.last_used = Instant::now();
                entry.adapter_bytes = adapter_bytes;
                Ok::<_, anyhow::Error>((weights, new_hash))
            } else {
                // Active checkpoint: apply the gradient directly and persist the new state.
                let (weights, new_hash, old_snapshot, adapter_bytes) = {
                    let trainer = entry.trainer.as_ref().ok_or_else(|| anyhow!("Cached checkpoint has no trainer"))?;
                    let named_grads = build_named_grads(trainer, averages)?;

                    let old_snapshot = CheckpointSnapshot {
                        weights: trainer.trainable_weights().context("Failed to snapshot active weights")?,
                        optimizer: trainer.clone_optimizer().context("Failed to snapshot active optimizer")?,
                    };

                    trainer.apply_gradients(&named_grads, 1e-5f32).context("Failed to apply averaged gradients")?;

                    let weights = trainer.save_weights_to_bytes().context("Failed to serialize updated weights")?;
                    let new_hash = <[u8; 32]>::from(blake3::hash(&weights));
                    let adapter_bytes = if is_lora { trainer.save_adapter_to_bytes().ok() } else { None };
                    (weights, new_hash, old_snapshot, adapter_bytes)
                };

                entry.snapshots.insert(old_head, old_snapshot);
                entry.head_hash = new_hash;
                entry.last_used = Instant::now();
                entry.adapter_bytes = adapter_bytes;
                Ok::<_, anyhow::Error>((weights, new_hash))
            }
        })
        .await
        .context("Gradient aggregation task panicked")??;

        self.finalize_new_head(&update.model_id, weights, old_head, new_hash, entry.clone()).await?;
        info!("FedAvg produced new active checkpoint for {}: hash {}", update.model_id, hex::encode(new_hash));
        Ok(Some(new_hash))
    }

    /// Store a new head as the active checkpoint, update lineage, and refresh the cache.
    async fn finalize_new_head(
        &self,
        model_id: &str,
        weights: Vec<u8>,
        old_head: [u8; 32],
        new_hash: [u8; 32],
        entry: Arc<Mutex<CachedCheckpoint>>,
    ) -> Result<()> {
        // Persist the new weights as the active checkpoint and also as a historical
        // snapshot keyed by its hash, so it can be re-served for block-height queries.
        self.storage
            .store_historical_weights(model_id, new_hash, &weights)
            .await
            .map_err(|e| anyhow!("Failed to store historical weights: {}", e))?;

        let files = self.storage.load_model_metadata(model_id).await.map_err(|e| anyhow!("Failed to load model metadata: {}", e))?;
        let new_files = RawModelFiles { config: files.config, tokenizer: files.tokenizer, weights };
        self.store_model_files(model_id, &new_files, ModelMetrics::default()).await?;

        // Track the lineage so we can rebase or accept stale-but-related blocks later.
        {
            let mut lineage = self.lineage.write().await;
            lineage.insert(new_hash, old_head);
        }

        // Insert a new cache entry keyed by the new head so that future gradients
        // for this checkpoint continue the same lineage.
        {
            let mut cache = self.checkpoint_cache.write().await;
            let mut order = self.cache_order.lock().await;
            cache.insert(new_hash, entry);
            order.push_back(new_hash);
            self.evict_lru_not_active(&mut cache, &mut order, new_hash);
        }

        Ok(())
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
                let entry = self.build_cached_checkpoint(model_id, active_hash, files).map_err(|e| {
                    warn!("Failed to build cached checkpoint for {} (hash {}): {:#}", model_id, hex::encode(active_hash), e);
                    e
                })?;
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

    /// Build a cached trainable checkpoint replica from raw model files.
    fn build_cached_checkpoint(
        &self,
        model_id: &str,
        active_hash: [u8; 32],
        files: RawModelFiles,
    ) -> Result<Arc<Mutex<CachedCheckpoint>>> {
        // MGM-1 is not aggregated by this node yet, but it still needs a
        // lightweight cache entry so checkpoint metadata can be served.
        if model_id.contains("mgm-1") {
            let entry = CachedCheckpoint {
                base_hash: active_hash,
                head_hash: active_hash,
                trainer: None,
                aggregator: FedAvgAggregator::new(self.fedavg_config.clone()),
                last_used: Instant::now(),
                model_id: model_id.to_string(),
                adapter_bytes: None,
                epoch_started: None,
                epoch_duration: self.epoch_duration,
                epoch_size_cap: self.epoch_size_cap,
                snapshots: HashMap::new(),
                pending_rebases: HashMap::new(),
                fedavg_config: self.fedavg_config.clone(),
            };
            return Ok(Arc::new(Mutex::new(entry)));
        }

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

        let mut snapshots = HashMap::new();
        let initial_weights = trainer.trainable_weights().context("Failed to snapshot initial trainable weights")?;
        let initial_optimizer = trainer.clone_optimizer().context("Failed to snapshot initial optimizer")?;
        snapshots.insert(active_hash, CheckpointSnapshot { weights: initial_weights, optimizer: initial_optimizer });

        let entry = CachedCheckpoint {
            base_hash,
            head_hash: active_hash,
            trainer: Some(trainer),
            aggregator: FedAvgAggregator::new(self.fedavg_config.clone()),
            last_used: Instant::now(),
            model_id: model_id.to_string(),
            adapter_bytes,
            epoch_started: None,
            epoch_duration: self.epoch_duration,
            epoch_size_cap: self.epoch_size_cap,
            snapshots,
            pending_rebases: HashMap::new(),
            fedavg_config: self.fedavg_config.clone(),
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

    fn build_test_tokenizer() -> Vec<u8> {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::tokenizer::AddedToken;

        let mut vocab: Vocab = Vocab::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("A".to_string(), 1);
        vocab.insert("T".to_string(), 2);
        vocab.insert("C".to_string(), 3);
        vocab.insert("G".to_string(), 4);
        vocab.insert("<mask>".to_string(), 5);

        // Add 2-mers so the test tokenizer is a realistic BPE/k-mer vocabulary.
        let bases = ['A', 'T', 'C', 'G'];
        let mut id = 6u32;
        for a in bases {
            for b in bases {
                let mut kmer = String::with_capacity(2);
                kmer.push(a);
                kmer.push(b);
                vocab.insert(kmer, id);
                id += 1;
            }
        }

        let bpe = BPE::new(vocab, vec![]);
        let mut tokenizer = tokenizers::Tokenizer::new(bpe);
        tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true), AddedToken::from("<pad>", true)]);

        serde_json::to_vec(&tokenizer).expect("failed to serialize test tokenizer")
    }

    fn insert_weight(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) {
        let n = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() + 0.001).collect();
        let t = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
        map.insert(name.to_string(), t);
    }

    fn build_tiny_dnabert2_files() -> RawModelFiles {
        // 22 = <pad>, A, T, C, G, <mask> (6) + 16 DNA 2-mers.
        let config = DnaBert2Config {
            vocab_size: 22,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 8,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".to_string(),
            position_embedding_type: "alibi".to_string(),
            alibi_starting_size: Some(16),
            tie_word_embeddings: true,
            pad_token_id: 0,
            mask_token_id: 5,
            bos_token_id: 1,
            eos_token_id: 2,
            num_labels: None,
        };

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        insert_weight(&mut tensors, "model.embeddings.word_embeddings.weight", &[config.vocab_size, config.hidden_size]);
        insert_weight(&mut tensors, "model.embeddings.token_type_embeddings.weight", &[config.type_vocab_size, config.hidden_size]);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.weight", &[config.hidden_size]);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.bias", &[config.hidden_size]);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.encoder.layer.{}", i);
            insert_weight(&mut tensors, &format!("{}.attention.self.query.weight", prefix), &[config.hidden_size, config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.attention.self.query.bias", prefix), &[config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.attention.self.key.weight", prefix), &[config.hidden_size, config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.attention.self.key.bias", prefix), &[config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.attention.self.value.weight", prefix), &[config.hidden_size, config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.attention.self.value.bias", prefix), &[config.hidden_size]);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.output.dense.weight", prefix),
                &[config.hidden_size, config.hidden_size],
            );
            insert_weight(&mut tensors, &format!("{}.attention.output.dense.bias", prefix), &[config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.weight", prefix), &[config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.bias", prefix), &[config.hidden_size]);
            insert_weight(
                &mut tensors,
                &format!("{}.mlp.up_proj.weight", prefix),
                &[config.intermediate_size * 2, config.hidden_size],
            );
            insert_weight(&mut tensors, &format!("{}.mlp.down_proj.weight", prefix), &[config.hidden_size, config.intermediate_size]);
            insert_weight(&mut tensors, &format!("{}.mlp.down_proj.bias", prefix), &[config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.mlp.layer_norm.weight", prefix), &[config.hidden_size]);
            insert_weight(&mut tensors, &format!("{}.mlp.layer_norm.bias", prefix), &[config.hidden_size]);
        }

        insert_weight(&mut tensors, "lm_head.transform.dense.weight", &[config.hidden_size, config.hidden_size]);
        insert_weight(&mut tensors, "lm_head.transform.dense.bias", &[config.hidden_size]);
        insert_weight(&mut tensors, "lm_head.transform.layer_norm.weight", &[config.hidden_size]);
        insert_weight(&mut tensors, "lm_head.transform.layer_norm.bias", &[config.hidden_size]);
        insert_weight(&mut tensors, "lm_head.bias", &[config.vocab_size]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let weights = std::fs::read(&path).unwrap();

        let config_bytes = serde_json::to_vec(&config).unwrap();
        let tokenizer_bytes = build_test_tokenizer();

        RawModelFiles { config: config_bytes, tokenizer: tokenizer_bytes, weights }
    }

    fn encrypted_update(model_id: &str, base: [u8; 32], payload: GradientPayload, key: &[u8; 32]) -> GradientUpdate {
        let plaintext = borsh::to_vec(&payload).unwrap();
        let encrypted_payload = model_crypto::encrypt(&plaintext, key).unwrap();

        let mut reconstructed = HashMap::with_capacity(payload.layer_gradients.len());
        for (name, layer) in &payload.layer_gradients {
            let flat = decompress_gradient_layer(layer).unwrap();
            let tensor = Tensor::from_vec(flat, layer.shape.clone(), &Device::Cpu).unwrap();
            reconstructed.insert(name.clone(), tensor);
        }
        let gradients_commitment = gradient_commitment(&reconstructed).unwrap();

        GradientUpdate {
            model_id: model_id.to_string(),
            base_checkpoint: base,
            encrypted_payload,
            participant_weight: 1.0,
            loss_before: 1.0,
            loss_after: 0.9,
            gradients_commitment,
            batch_indices: vec![0, 1, 2],
            batch_id: 1,
            learning_rate: 0.001,
            genome_merkle_root: [0u8; 32],
            genome_slices: Vec::new(),
            compute_time_ms: 100,
        }
    }

    #[tokio::test]
    async fn test_delta_rebase_produces_new_active_checkpoint() {
        // Force single-gradient epochs so the test does not have to wait.
        std::env::set_var("XENO_EPOCH_DURATION_SECS", "0");
        std::env::set_var("XENO_FEDAVG_EPOCH_SIZE", "1");

        let dir = tempfile::tempdir().unwrap();
        let key = [0u8; 32];
        let manager = ModelManager::new_with_key(dir.path().to_string_lossy().to_string(), key, None, None, None, None).await.unwrap();
        let model_id = "dnabert2-tiny";

        let files = build_tiny_dnabert2_files();
        manager.store_model_files(model_id, &files, ModelMetrics::default()).await.unwrap();
        manager.load_model(model_id).await.unwrap();

        let base = manager.active_hash(model_id).await.unwrap();

        // Active update: produces H1.
        let payload = GradientPayload {
            layer_gradients: HashMap::from([(
                "lm_head.bias".to_string(),
                GradientLayer { values: vec![0.1; 22], shape: vec![22], indices: vec![] },
            )]),
        };
        let h1 = manager
            .submit_gradients(&encrypted_update(model_id, base, payload.clone(), &key))
            .await
            .unwrap()
            .expect("active update should produce a new checkpoint");
        assert_ne!(h1, base);
        assert_eq!(manager.active_hash(model_id).await.unwrap(), h1);

        // Stale-but-related update: computed against base A, rebased onto active H1.
        let h2 = manager
            .submit_gradients(&encrypted_update(model_id, base, payload, &key))
            .await
            .unwrap()
            .expect("stale update should be rebased onto active checkpoint");
        assert_ne!(h2, h1);
        assert_ne!(h2, base);
        assert_eq!(manager.active_hash(model_id).await.unwrap(), h2);

        // Ensure the resulting checkpoint can be loaded without corruption.
        let (checkpoint, loaded) = manager.get_model_checkpoint(model_id).await.unwrap();
        assert_eq!(checkpoint.weights_hash, h2);
        assert!(!loaded.weights.is_empty());
        assert!(loaded.weights.len() > 8);
        assert_eq!(loaded.weights[8], b'{'); // SafeTensors header starts with JSON
    }
}
