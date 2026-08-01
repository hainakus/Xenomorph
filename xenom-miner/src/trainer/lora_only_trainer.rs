//! LoRA-only trainer that drives the full secure distribution loop.
//!
//! This is an integration spike: it fetches a `TrainingArtifact` from the
//! orchestrator, loads the frozen LM head base weights with a LoRA adapter,
//! requests attested hidden states for each batch, trains the adapter, and
//! submits an encrypted LoRA delta.
//!
//! For this spike the artifact and hidden states are not encrypted.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::data::MlmBatchGenerator;
use crate::lora::LoraConfig;
use crate::model::DnaBert2Config;
use crate::rpc::client::XenomRpcClient;
use crate::rpc::messages::{AttestedForwardRequest, GenomeTrainingBatchMsg, SubmitLoRAUpdate, TrainingArtifact};
use crate::tokenizer::DnaTokenizer;
use crate::trainer::lora_lm_head::LoraLmHead;
use crate::trainer::ManualAdamW;

/// Drives a single LoRA-only training round.
pub struct LoraOnlyTrainer {
    client: XenomRpcClient,
    device: Device,
    learning_rate: f64,
    local_steps: usize,
    lora_config: LoraConfig,
    /// Cached artifact data across rounds.
    base: Option<ArtifactBase>,
}

struct ArtifactBase {
    model_id: String,
    base_checkpoint: [u8; 32],
    config: DnaBert2Config,
    tokenizer: DnaTokenizer,
    base_weights: HashMap<String, Tensor>,
}

impl LoraOnlyTrainer {
    pub fn new(client: XenomRpcClient, learning_rate: f64, local_steps: usize, lora_config: LoraConfig) -> Self {
        Self { client, device: Device::Cpu, learning_rate, local_steps, lora_config, base: None }
    }

    /// Return the active base checkpoint, if an artifact has been loaded.
    pub fn current_base_checkpoint(&self) -> Option<[u8; 32]> {
        self.base.as_ref().map(|b| b.base_checkpoint)
    }

    /// Clear the cached artifact so the next round fetches it again.
    pub fn clear_artifact(&mut self) {
        self.base = None;
    }

    /// Train one genome-backed batch and return the final loss and LoRA delta.
    ///
    /// This fetches (or reuses) the training artifact, tokenizes the provided DNA
    /// sequences into an MLM batch, requests attested hidden states, trains, and
    /// submits a `LoRAUpdate`.
    pub async fn train_genome_round(&mut self, msg: &GenomeTrainingBatchMsg, miner_public_key: [u8; 33]) -> Result<(f64, Vec<u8>)> {
        let model_id = msg.batch.model_id.clone();
        let base_checkpoint = msg.base_checkpoint;
        let seed = msg.base_checkpoint;

        // 1. Ensure we have the artifact for this base checkpoint.
        self.ensure_artifact(&model_id, base_checkpoint, miner_public_key).await?;

        // 2. Tokenize the sequences into an MLM batch.
        let base = self.base.as_ref().unwrap();
        let tokenizer = base.tokenizer.clone();
        let batch_size = msg.sequences.len();
        if batch_size == 0 {
            bail!("Empty genome batch");
        }
        let generator = MlmBatchGenerator::new(tokenizer, base.config.max_position_embeddings)
            .with_mask_prob(0.15)
            .with_span_len(6.min(base.config.max_position_embeddings));
        let source_indices: Vec<u64> = msg.batch.data_indices.iter().map(|s| s.chunk_idx).collect();
        let mlm = generator.generate_from_sequences_with_indices(&msg.sequences, &seed, msg.batch.batch_id, Some(&source_indices))?;

        // 3. Reshape into per-row 2D vectors as expected by `train_round_tensors`.
        let input_ids: Vec<Vec<u32>> = mlm.input_ids.chunks(mlm.seq_len).map(|c| c.to_vec()).collect();
        let attention_mask: Vec<Vec<u32>> = mlm.attention_mask.chunks(mlm.seq_len).map(|c| c.to_vec()).collect();
        let labels: Vec<Vec<u32>> = mlm.labels.chunks(mlm.seq_len).map(|c| c.to_vec()).collect();
        let mask: Vec<Vec<u8>> = mlm.mask.chunks(mlm.seq_len).map(|c| c.to_vec()).collect();

        // 4. Run the tensor-level round (which also reuses the loaded artifact).
        self.train_round_tensors(&model_id, base_checkpoint, miner_public_key, input_ids, attention_mask, labels, mask).await
    }

    /// Run one training round from pre-tokenized tensors.
    pub async fn train_round(
        &mut self,
        model_id: &str,
        base_checkpoint: [u8; 32],
        miner_public_key: [u8; 33],
        input_ids: Vec<Vec<u32>>,
        attention_mask: Vec<Vec<u32>>,
        labels: Vec<Vec<u32>>,
        mask: Vec<Vec<u8>>,
    ) -> Result<(f64, Vec<u8>)> {
        self.ensure_artifact(model_id, base_checkpoint, miner_public_key).await?;
        self.train_round_tensors(model_id, base_checkpoint, miner_public_key, input_ids, attention_mask, labels, mask).await
    }

    async fn ensure_artifact(&mut self, model_id: &str, base_checkpoint: [u8; 32], miner_public_key: [u8; 33]) -> Result<()> {
        if let Some(base) = self.base.as_ref() {
            if base.model_id == model_id && base.base_checkpoint == base_checkpoint {
                return Ok(());
            }
        }

        let artifact = self
            .client
            .get_training_artifact(model_id, base_checkpoint, Some(base_checkpoint), miner_public_key)
            .await
            .with_context(|| format!("Failed to fetch training artifact for {}", model_id))?;

        if artifact.encrypted {
            bail!("Training artifact is encrypted, but decryption is not yet implemented in the spike");
        }

        let (config, tokenizer, base_weights) = Self::load_artifact(&artifact)?;
        self.base = Some(ArtifactBase { model_id: model_id.to_string(), base_checkpoint, config, tokenizer, base_weights });
        Ok(())
    }

    async fn train_round_tensors(
        &mut self,
        model_id: &str,
        base_checkpoint: [u8; 32],
        _miner_public_key: [u8; 33],
        input_ids: Vec<Vec<u32>>,
        attention_mask: Vec<Vec<u32>>,
        labels: Vec<Vec<u32>>,
        mask: Vec<Vec<u8>>,
    ) -> Result<(f64, Vec<u8>)> {
        let base = self.base.as_ref().ok_or_else(|| anyhow!("No artifact loaded"))?;

        // 2. Build the LoRA LM head.
        let head = LoraLmHead::load(&base.config, &base.base_weights, &self.lora_config, &self.device, DType::F32)?;

        // 3. Request attested hidden states.
        let forward_req = AttestedForwardRequest {
            model_id: model_id.to_string(),
            base_checkpoint,
            input_ids: input_ids.clone(),
            attention_mask: attention_mask.clone(),
            labels: labels.clone(),
            mask: mask.clone(),
        };
        let forward = self.client.attested_forward(forward_req).await.with_context(|| "AttestedForward request failed")?;

        // 4. Deserialize hidden states [batch, seq, hidden_size].
        let batch_size = input_ids.len();
        if batch_size == 0 {
            bail!("Empty batch");
        }
        let seq_len = input_ids[0].len();
        let hidden_size = base.config.hidden_size;

        let hidden_floats: Vec<f32> =
            forward.hidden_states.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let hidden_states = Tensor::from_vec(hidden_floats, (batch_size, seq_len, hidden_size), &self.device)?;

        let labels_t = Tensor::from_vec(labels.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &self.device)?;
        let mask_t = Tensor::from_vec(mask.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &self.device)?;

        // 5. Train the LoRA adapter.
        let mut optimizer = ManualAdamW::new(self.learning_rate);
        let mut last_loss = 0.0;
        for _ in 0..self.local_steps {
            last_loss = head.train_step(&hidden_states, &labels_t, &mask_t, &mut optimizer)?;
        }

        // 6. Save the adapter and submit the update.
        let delta = head.save_adapter()?;

        let update = SubmitLoRAUpdate {
            model_id: model_id.to_string(),
            base_checkpoint,
            lora_delta: delta.clone(),
            gradient_commitment: blake3_hash(&delta),
            participant_weight: 1.0,
            miner_address: String::new(),
        };
        let _new_checkpoint = self.client.submit_lora_update(update).await?;

        Ok((last_loss, delta))
    }

    fn load_artifact(artifact: &TrainingArtifact) -> Result<(DnaBert2Config, DnaTokenizer, HashMap<String, Tensor>)> {
        let config = DnaBert2Config::from_bytes(&artifact.config)?;
        let tokenizer = DnaTokenizer::from_bytes(&artifact.tokenizer)?;

        let device = Device::Cpu;
        let loaded = candle_core::safetensors::load_buffer(&artifact.artifact, &device)
            .map_err(|e| anyhow!("Failed to load artifact safetensors: {}", e))?;

        let mut base_weights = HashMap::new();
        for (name, tensor) in loaded.iter() {
            base_weights.insert(name.clone(), tensor.clone());
        }

        Ok((config, tokenizer, base_weights))
    }
}

fn blake3_hash(data: &[u8]) -> [u8; 32] {
    let hash = blake3::hash(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}
