//! Unified training coordinator inside the full node.
//!
//! The coordinator replaces the standalone `seed-node` for devnet deployments:
//! it stores the active model, serves genome batches to miners, accepts completed
//! training blocks, validates the training proof, builds a Kaspa block and mines it.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use borsh::{to_vec, BorshSerialize};
use kaspa_addresses::{Address, Prefix};
use kaspa_consensus_core::header::Header;
use kaspa_consensus_core::network::NetworkType;
use kaspa_consensus_core::pow::{DifficultyTarget, ModelId, PublicInputs, TrainingProof as ConsensusTrainingProof, ZKProof};
use kaspa_core::{info, warn};
use kaspa_hashes::Hash;
use kaspa_pow::genome_pow::GenomeDatasetLoader;
use kaspa_pow::State as PowState;
use kaspa_rpc_core::api::rpc::RpcApi;
use kaspa_rpc_core::{GetBlockTemplateRequest, SubmitBlockReport, SubmitBlockRequest};
use kaspa_rpc_service::service::RpcCoreService;
use seed_node::genome::{GenomeBatchGenerator, GenomeStorage};
use seed_node::model::manager::ModelManager;
use seed_node::model::storage::ModelStorage;
use seed_node::rpc::messages::{
    GenomeTrainingBatchMsg, GetGenomeTrainingBatch, GradientUpdate, ModelCheckpoint as RpcModelCheckpoint,
    ModelCheckpointInfo as RpcModelCheckpointInfo, RpcResponse, TrainingBatch, TrainingBlock,
};
use tokio::sync::RwLock;

/// Compact training summary embedded into the coinbase extra-data payload.
#[derive(Debug, Clone, BorshSerialize)]
pub struct CoinbaseExtraData {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: [u8; 32],
}

/// Lightweight handle to the training coordinator. Clone is cheap (all state is Arc-wrapped).
#[derive(Clone)]
pub struct Coordinator {
    inner: Arc<CoordinatorInner>,
}

struct CoordinatorInner {
    network_type: NetworkType,
    active_model_id: String,
    active_weights_hash: RwLock<Option<Hash>>,
    difficulty: DifficultyTarget,

    model_manager: Arc<ModelManager>,
    genome_storage: Arc<RwLock<GenomeStorage>>,
    genome_source_url: String,
    genome_file: Option<PathBuf>,

    rpc_core_service: Arc<RpcCoreService>,
    genome_fragment_size_bytes: u32,
    genome_pow_activation_daa_score: u64,

    current_epoch: AtomicU64,
}

impl Coordinator {
    pub async fn new(
        network_type: NetworkType,
        active_model_id: String,
        models_dir: PathBuf,
        genome_cache_dir: PathBuf,
        genome_file: Option<PathBuf>,
        genome_source_url: String,
        rpc_core_service: Arc<RpcCoreService>,
        genome_fragment_size_bytes: u32,
        genome_pow_activation_daa_score: u64,
    ) -> Result<Self> {
        tokio::fs::create_dir_all(&models_dir).await?;
        tokio::fs::create_dir_all(&genome_cache_dir).await?;

        // Use a stable key for the local model cache so restarts do not force a re-download.
        // The same derivation is used by the seed-node so they can share a model cache directory.
        let encryption_key = ModelStorage::derive_encryption_key();

        let model_manager = Arc::new(ModelManager::new_with_key(models_dir.to_string_lossy().to_string(), encryption_key).await?);

        // Pre-download the active model before accepting miner connections. This avoids the
        // 30s RPC request timeout in xenom-miner while the full node is still downloading.
        info!("Pre-downloading active model {} ...", active_model_id);
        model_manager.ensure_model_downloaded(&active_model_id).await?;

        let mut genome_storage = GenomeStorage::new(&genome_cache_dir).await?;
        let mut genome_file_path = None;
        if let Some(path) = genome_file {
            let archive = seed_node::genome::archive::GenomeArchive::load(&path)
                .with_context(|| format!("Failed to load genome archive from {:?}", path))?;
            let merkle_root = archive.header.merkle_root;
            genome_storage.load_from_path(merkle_root, &path).await?;
            info!("Loaded genome archive {:?} with merkle root {}", path, hex::encode(merkle_root));
            genome_file_path = Some(path);
        }

        let difficulty = DifficultyTarget { min_improvement: -1.0, max_loss_after: f64::MAX };

        let coordinator = Self {
            inner: Arc::new(CoordinatorInner {
                network_type,
                active_model_id: active_model_id.clone(),
                active_weights_hash: RwLock::new(None),
                difficulty,
                model_manager,
                genome_storage: Arc::new(RwLock::new(genome_storage)),
                genome_source_url,
                genome_file: genome_file_path,
                rpc_core_service,
                genome_fragment_size_bytes,
                genome_pow_activation_daa_score,
                current_epoch: AtomicU64::new(0),
            }),
        };

        // Compute and cache the weights hash now so subsequent miner requests are fast.
        let _ = coordinator.active_weights_hash().await?;

        Ok(coordinator)
    }

    /// Return the active model's weights hash, downloading the model if necessary.
    /// The value is cached after the first call.
    pub async fn active_weights_hash(&self) -> Result<Hash> {
        {
            let cached = self.inner.active_weights_hash.read().await;
            if let Some(hash) = *cached {
                return Ok(hash);
            }
        }

        let model_id = &self.inner.active_model_id;
        info!("Downloading/loading active model {} for weights hash", model_id);

        // Prefer the lighter `load_model` path; fall back to the full checkpoint files.
        let weights_hash = match self.inner.model_manager.load_model(model_id).await {
            Ok(info) => info.checkpoint.weights_hash,
            Err(_) => {
                self.inner.model_manager.ensure_model_downloaded(model_id).await?;
                let (checkpoint, _) = self.inner.model_manager.get_model_checkpoint(model_id).await?;
                checkpoint.weights_hash
            }
        };

        let hash = Hash::from_bytes(weights_hash);
        *self.inner.active_weights_hash.write().await = Some(hash);
        info!("Active model weights hash: {}", hex::encode(weights_hash));
        Ok(hash)
    }

    /// Hand out a non-genome (synthetic) training batch.
    pub async fn get_training_batch(&self, model_id: String) -> RpcResponse {
        if model_id != self.inner.active_model_id {
            return RpcResponse::Error(format!("Unknown model id {} (active is {})", model_id, self.inner.active_model_id));
        }

        let base_checkpoint = match self.active_weights_hash().await {
            Ok(hash) => hash.as_bytes(),
            Err(e) => return RpcResponse::Error(format!("Failed to load active model: {}", e)),
        };

        let batch_id = self.inner.current_epoch.load(Ordering::Relaxed);
        RpcResponse::TrainingBatch(Some(TrainingBatch {
            batch_id,
            model_id,
            base_checkpoint,
            data_indices: vec![],
            target_improvement: 0.01,
            learning_rate: 0.01,
        }))
    }

    /// Hand out a genome-backed DNABERT-2 training batch.
    pub async fn get_genome_training_batch(&self, request: GetGenomeTrainingBatch) -> RpcResponse {
        if request.model_id != self.inner.active_model_id {
            return RpcResponse::Error(format!("Unknown model id {} (active is {})", request.model_id, self.inner.active_model_id));
        }

        let base_checkpoint = match self.active_weights_hash().await {
            Ok(hash) => hash.as_bytes(),
            Err(e) => return RpcResponse::Error(format!("Failed to load active model: {}", e)),
        };

        let archive = {
            let mut storage = self.inner.genome_storage.write().await;
            match storage.get_or_load(request.genome_merkle_root, &self.inner.genome_source_url).await {
                Ok(archive) => archive,
                Err(e) => return RpcResponse::Error(format!("Failed to load genome archive: {}", e)),
            }
        };

        let mut seed = [0u8; 32];
        seed.copy_from_slice(&request.genome_merkle_root);
        // XOR the first 8 bytes with the current epoch so each accepted training
        // block produces a different genome batch instead of repeating the same slice.
        let epoch = self.inner.current_epoch.load(Ordering::Relaxed);
        let epoch_bytes = epoch.to_le_bytes();
        for i in 0..8 {
            seed[i] ^= epoch_bytes[i];
        }
        let mut generator = GenomeBatchGenerator::new(archive, seed);
        let mut batch = generator.generate_batch(request.preferred_batch_size.max(1), 128);
        batch.model_id = request.model_id;

        let sequences = generator.extract_sequences(&batch);
        RpcResponse::GenomeTrainingBatch(GenomeTrainingBatchMsg { batch, sequences, base_checkpoint })
    }

    /// Return lightweight checkpoint metadata to the miner so it can check its
    /// local cache without downloading the full weights.
    pub async fn get_model_checkpoint_info(&self, model_id: String) -> RpcResponse {
        if model_id != self.inner.active_model_id {
            return RpcResponse::Error(format!("Unknown model id {} (active is {})", model_id, self.inner.active_model_id));
        }

        let base_checkpoint = match self.active_weights_hash().await {
            Ok(hash) => hash.as_bytes(),
            Err(e) => return RpcResponse::Error(format!("Failed to load active model: {}", e)),
        };

        RpcResponse::ModelCheckpointInfo(RpcModelCheckpointInfo { model_id, base_checkpoint })
    }

    /// Return the raw model checkpoint files to the miner.
    pub async fn get_model_checkpoint(&self, model_id: String) -> RpcResponse {
        if model_id != self.inner.active_model_id {
            return RpcResponse::Error(format!("Unknown model id {} (active is {})", model_id, self.inner.active_model_id));
        }

        if let Err(e) = self.inner.model_manager.ensure_model_downloaded(&model_id).await {
            return RpcResponse::Error(format!("Failed to download model: {}", e));
        }

        let (checkpoint, files) = match self.inner.model_manager.get_encrypted_model_checkpoint(&model_id).await {
            Ok(cp) => cp,
            Err(e) => return RpcResponse::Error(format!("Failed to load model checkpoint: {}", e)),
        };

        RpcResponse::ModelCheckpoint(RpcModelCheckpoint {
            model_id,
            base_checkpoint: checkpoint.weights_hash,
            config: files.config,
            tokenizer: files.tokenizer,
            weights: files.weights,
            encrypted: true,
        })
    }

    /// Accept a completed training block, validate it, build a Kaspa block and submit it.
    pub async fn submit_block(&self, block: TrainingBlock) -> RpcResponse {
        // Address prefix check.
        let address = match Address::try_from(block.miner_address.as_str()) {
            Ok(addr) => addr,
            Err(e) => {
                warn!("Rejecting training block: invalid miner address {}: {}", block.miner_address, e);
                return RpcResponse::Error(format!("Invalid miner address: {}", e));
            }
        };
        let expected_prefix = Prefix::from(self.inner.network_type);
        if address.prefix != expected_prefix {
            warn!("Rejecting training block: address prefix {} does not match network {:?}", address.prefix, self.inner.network_type);
            return RpcResponse::Error(format!("Address prefix does not match network {:?}", self.inner.network_type));
        }

        // Active model validation.
        if block.model_id != self.inner.active_model_id {
            warn!("Rejecting training block: model id {} != active {}", block.model_id, self.inner.active_model_id);
            return RpcResponse::Error(format!("Model {} is not the active model", block.model_id));
        }

        let active_weights_hash = match self.active_weights_hash().await {
            Ok(h) => h,
            Err(e) => return RpcResponse::Error(format!("Failed to load active model: {}", e)),
        };

        let miner_proof = &block.training_proof;
        let base_checkpoint = Hash::from_bytes(miner_proof.base_checkpoint);
        if base_checkpoint != active_weights_hash {
            warn!("Rejecting training block: base checkpoint {} != active weights hash {}", base_checkpoint, active_weights_hash);
            return RpcResponse::Error("Base checkpoint does not match active model weights hash".to_string());
        }

        // Difficulty target (lazy/optimistic: allow loss to rise by up to 1.0 on devnet).
        let consensus_proof = ConsensusTrainingProof {
            model_id: ModelId(block.model_id.clone()),
            base_checkpoint,
            loss_before: miner_proof.loss_before,
            loss_after: miner_proof.loss_after,
            gradients_commitment: Hash::from_bytes(miner_proof.gradients_commitment),
            zk_proof: ZKProof {
                proof_data: miner_proof.zk_proof.clone(),
                public_inputs: PublicInputs {
                    model_hash: Hash::default(),
                    input_hash: Hash::default(),
                    output_gradients_hash: Hash::from_bytes(miner_proof.gradients_commitment),
                    loss_before: miner_proof.loss_before,
                    loss_after: miner_proof.loss_after,
                },
            },
            batch_indices: miner_proof.batch_indices.clone(),
        };

        if !consensus_proof.verify(&self.inner.difficulty) {
            warn!(
                "Rejecting training block: training proof does not meet difficulty target (loss_before={}, loss_after={}, min_improvement={})",
                miner_proof.loss_before,
                miner_proof.loss_after,
                self.inner.difficulty.min_improvement
            );
            return RpcResponse::Error("Training proof does not meet difficulty target".to_string());
        }

        // Build and mine the block.
        let extra_data = CoinbaseExtraData {
            model_id: block.model_id.clone(),
            base_checkpoint: active_weights_hash.as_bytes(),
            loss_before: miner_proof.loss_before,
            loss_after: miner_proof.loss_after,
            gradients_commitment: miner_proof.gradients_commitment,
        };
        let extra_data_bytes = match to_vec(&extra_data) {
            Ok(bytes) => bytes,
            Err(e) => return RpcResponse::Error(format!("Failed to serialize coinbase extra data: {}", e)),
        };

        let template_request = GetBlockTemplateRequest::new(address, extra_data_bytes);
        let template = match self.inner.rpc_core_service.get_block_template_call(None, template_request).await {
            Ok(t) => t,
            Err(e) => {
                warn!("get_block_template failed for training block: {}", e);
                return RpcResponse::Error(format!("get_block_template failed: {}", e));
            }
        };

        if !template.is_synced {
            return RpcResponse::Error("Node is not synced; cannot produce training block".to_string());
        }

        let mut raw_block = template.block;
        let mut header: Header = raw_block.header.into();

        let found_nonce = if header.daa_score >= self.inner.genome_pow_activation_daa_score {
            let state = kaspa_pow::genome_pow_state(&header, self.inner.genome_fragment_size_bytes);
            let mut nonce = 0u64;

            if let Some(ref path) = self.inner.genome_file {
                // Real packed genome dataset is available: mine with the same memory-hard
                // genome_mix_hash path consensus will use to validate the block.
                match kaspa_pow::genome_file::FileGenomeLoader::open(path, self.inner.genome_fragment_size_bytes, false) {
                    Ok(loader) => {
                        let packed = loader.packed_dataset().unwrap_or(&[]);
                        loop {
                            if state.check_pow_memory_hard(nonce, packed).0 {
                                break nonce;
                            }
                            nonce = nonce.wrapping_add(1);
                            if nonce == 0 {
                                return RpcResponse::Error("Failed to solve memory-hard Genome PoW for training block".to_string());
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to open genome file {:?} for mining: {}; falling back to synthetic fragment", path, e);
                        let loader =
                            kaspa_pow::genome_pow::SyntheticLoader::new(self.inner.genome_fragment_size_bytes, header.epoch_seed);
                        loop {
                            let fragment_idx = state.fragment_index_for(nonce);
                            let Some(fragment) = loader.load_fragment(fragment_idx) else {
                                return RpcResponse::Error(format!("Failed to synthesize genome fragment {}", fragment_idx));
                            };
                            if state.check_pow_with_fragment(nonce, &fragment).0 {
                                break nonce;
                            }
                            nonce = nonce.wrapping_add(1);
                            if nonce == 0 {
                                return RpcResponse::Error("Failed to solve Genome PoW for training block".to_string());
                            }
                        }
                    }
                }
            } else {
                let loader = kaspa_pow::genome_pow::SyntheticLoader::new(self.inner.genome_fragment_size_bytes, header.epoch_seed);
                loop {
                    let fragment_idx = state.fragment_index_for(nonce);
                    let Some(fragment) = loader.load_fragment(fragment_idx) else {
                        return RpcResponse::Error(format!("Failed to synthesize genome fragment {}", fragment_idx));
                    };
                    if state.check_pow_with_fragment(nonce, &fragment).0 {
                        break nonce;
                    }
                    nonce = nonce.wrapping_add(1);
                    if nonce == 0 {
                        return RpcResponse::Error("Failed to solve Genome PoW for training block".to_string());
                    }
                }
            }
        } else {
            let state = PowState::new(&header);
            let mut nonce = 0u64;
            loop {
                if state.check_pow(nonce).0 {
                    break nonce;
                }
                nonce = nonce.wrapping_add(1);
                if nonce == 0 {
                    return RpcResponse::Error("Failed to solve PoW for training block".to_string());
                }
            }
        };

        header.nonce = found_nonce;
        header.finalize();
        let block_hash: [u8; 32] = header.hash.as_bytes();
        raw_block.header = header.into();

        let submit_request = SubmitBlockRequest { block: raw_block, allow_non_daa_blocks: false };
        let submit_response = match self.inner.rpc_core_service.submit_block_call(None, submit_request).await {
            Ok(resp) => resp,
            Err(e) => {
                warn!("submit_block failed for training block: {}", e);
                return RpcResponse::Error(format!("submit_block failed: {}", e));
            }
        };

        let accepted = matches!(submit_response.report, SubmitBlockReport::Success);
        if accepted {
            info!("Accepted training block {} (hash {})", block.header.block_number, Hash::from(block_hash));
            self.inner.current_epoch.fetch_add(1, Ordering::Relaxed);
        } else {
            warn!("Consensus rejected training block {}: {:?}", block.header.block_number, submit_response.report);
        }

        RpcResponse::BlockHash(block_hash)
    }

    /// Submit a gradient update for FedAvg aggregation and, if enough participants
    /// have contributed, apply the averaged update to the active model.
    pub async fn submit_gradients(&self, update: &GradientUpdate) -> Result<Option<[u8; 32]>> {
        if update.model_id != self.inner.active_model_id {
            return Err(anyhow::anyhow!(
                "Gradient update for {} does not match active model {}",
                update.model_id,
                self.inner.active_model_id
            ));
        }

        let new_checkpoint = self.inner.model_manager.submit_gradients(update).await?;

        // Update the cached active weights hash so the next miner round uses the new checkpoint.
        if let Some(hash) = new_checkpoint {
            let hash = Hash::from_bytes(hash);
            let mut cached = self.inner.active_weights_hash.write().await;
            *cached = Some(hash);
        }

        Ok(new_checkpoint)
    }
}
