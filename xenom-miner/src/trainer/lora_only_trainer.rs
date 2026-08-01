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

use crate::lora::LoraConfig;
use crate::model::DnaBert2Config;
use crate::rpc::client::XenomRpcClient;
use crate::rpc::messages::{AttestedForwardRequest, SubmitLoRAUpdate, TrainingArtifact};
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
}

impl LoraOnlyTrainer {
    pub fn new(client: XenomRpcClient, learning_rate: f64, local_steps: usize, lora_config: LoraConfig) -> Self {
        Self { client, device: Device::Cpu, learning_rate, local_steps, lora_config }
    }

    /// Run one training round.
    ///
    /// `base_checkpoint` is the active checkpoint id.  `miner_public_key` is the
    /// miner's public key (currently only used to fill the request; encryption is
    /// not yet wired).  `input_ids`, `attention_mask`, `labels`, and `mask` are
    /// a single pre-tokenized batch.
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
        // 1. Fetch the training artifact (LM head base weights + LoRA seed).
        let artifact = self
            .client
            .get_training_artifact(model_id, base_checkpoint, Some(base_checkpoint), miner_public_key)
            .await
            .with_context(|| format!("Failed to fetch training artifact for {}", model_id))?;

        if artifact.encrypted {
            bail!("Training artifact is encrypted, but decryption is not yet implemented in the spike");
        }

        let (config, _tokenizer, base_weights) = Self::load_artifact(&artifact)?;

        // 2. Build the LoRA LM head.
        let head = LoraLmHead::load(&config, &base_weights, &self.lora_config, &self.device, DType::F32)?;

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
        let hidden_size = config.hidden_size;

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
