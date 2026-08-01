//! LoRA-only trainer that drives the full secure distribution loop.
//!
//! This is an integration spike: it fetches a `TrainingArtifact` from the
//! orchestrator, loads the frozen LM head base weights with a LoRA adapter,
//! requests attested hidden states for each batch, trains the adapter, and
//! submits an encrypted LoRA delta.
//!
//! The artifact is encrypted with an ephemeral session key derived from ECDH
//! between an ephemeral orchestrator key and the miner's secp256k1 public key.
//! The miner uses its secp256k1 secret to decrypt and verify the signature.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use model_crypto::artifact_sign::ArtifactVerifier;
use model_crypto::session;
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

use crate::data::MlmBatchGenerator;
use crate::lora::LoraConfig;
use crate::model::DnaBert2Config;
use crate::rpc::client::XenomRpcClient;
use crate::rpc::messages::{
    AttestedForwardRequest, AttestedForwardResponse, GenomeTrainingBatchMsg, SubmitLoRAUpdate, TrainingArtifact,
};
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
    /// Miner secp256k1 secret used to decrypt artifacts and hidden states.
    miner_secret: SecretKey,
    /// Miner secp256k1 public key sent to the orchestrator.
    miner_public_key: [u8; 33],
    /// Session info from the most recent AttestedForward, used to encrypt the LoRA delta.
    last_forward_session: Option<ForwardSession>,
}

/// Session info needed to re-derive the AttestedForward session key on the miner side.
struct ForwardSession {
    ephemeral_public_key: [u8; 33],
    session_nonce: [u8; 12],
}

struct ArtifactBase {
    model_id: String,
    base_checkpoint: [u8; 32],
    config: DnaBert2Config,
    tokenizer: DnaTokenizer,
    base_weights: HashMap<String, Tensor>,
}

impl LoraOnlyTrainer {
    pub fn new(
        client: XenomRpcClient,
        learning_rate: f64,
        local_steps: usize,
        lora_config: LoraConfig,
        miner_secret: SecretKey,
    ) -> Self {
        let secp = Secp256k1::new();
        let miner_public_key = PublicKey::from_secret_key(&secp, &miner_secret).serialize();
        Self {
            client,
            device: Device::Cpu,
            learning_rate,
            local_steps,
            lora_config,
            base: None,
            miner_secret,
            miner_public_key,
            last_forward_session: None,
        }
    }

    /// Return the active base checkpoint, if an artifact has been loaded.
    pub fn current_base_checkpoint(&self) -> Option<[u8; 32]> {
        self.base.as_ref().map(|b| b.base_checkpoint)
    }

    /// Clear the cached artifact so the next round fetches it again.
    pub fn clear_artifact(&mut self) {
        self.base = None;
    }

    /// Return the miner public key used for ECDH session key derivation.
    pub fn miner_public_key(&self) -> [u8; 33] {
        self.miner_public_key
    }

    /// Train one genome-backed batch and return the final loss and LoRA delta.
    ///
    /// This fetches (or reuses) the training artifact, tokenizes the provided DNA
    /// sequences into an MLM batch, requests attested hidden states, trains, and
    /// submits a `LoRAUpdate`.
    pub async fn train_genome_round(&mut self, msg: &GenomeTrainingBatchMsg) -> Result<(f64, Vec<u8>)> {
        let model_id = msg.batch.model_id.clone();
        let base_checkpoint = msg.base_checkpoint;
        let seed = msg.base_checkpoint;

        // 1. Ensure we have the artifact for this base checkpoint.
        self.ensure_artifact(&model_id, base_checkpoint).await?;

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
        self.train_round_tensors(&model_id, base_checkpoint, input_ids, attention_mask, labels, mask).await
    }

    /// Run one training round from pre-tokenized tensors.
    pub async fn train_round(
        &mut self,
        model_id: &str,
        base_checkpoint: [u8; 32],
        input_ids: Vec<Vec<u32>>,
        attention_mask: Vec<Vec<u32>>,
        labels: Vec<Vec<u32>>,
        mask: Vec<Vec<u8>>,
    ) -> Result<(f64, Vec<u8>)> {
        self.ensure_artifact(model_id, base_checkpoint).await?;
        self.train_round_tensors(model_id, base_checkpoint, input_ids, attention_mask, labels, mask).await
    }

    async fn ensure_artifact(&mut self, model_id: &str, base_checkpoint: [u8; 32]) -> Result<()> {
        if let Some(base) = self.base.as_ref() {
            if base.model_id == model_id && base.base_checkpoint == base_checkpoint {
                return Ok(());
            }
        }

        let artifact = self
            .client
            .get_training_artifact(model_id, base_checkpoint, Some(base_checkpoint), self.miner_public_key)
            .await
            .with_context(|| format!("Failed to fetch training artifact for {}", model_id))?;

        let (config, tokenizer, base_weights) = self.load_artifact(&artifact)?;
        self.base = Some(ArtifactBase { model_id: model_id.to_string(), base_checkpoint, config, tokenizer, base_weights });
        Ok(())
    }

    async fn train_round_tensors(
        &mut self,
        model_id: &str,
        base_checkpoint: [u8; 32],
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
            miner_public_key: self.miner_public_key,
        };
        let forward = self.client.attested_forward(forward_req).await.with_context(|| "AttestedForward request failed")?;

        // 4. Decrypt and verify the attested hidden states.
        let (hidden_states_bytes, forward_session) = self.decrypt_and_verify_forward(&forward, base_checkpoint)?;
        self.last_forward_session = Some(forward_session);

        // 5. Deserialize hidden states [batch, seq, hidden_size].
        let batch_size = input_ids.len();
        if batch_size == 0 {
            bail!("Empty batch");
        }
        let seq_len = input_ids[0].len();
        let hidden_size = base.config.hidden_size;

        let hidden_floats: Vec<f32> =
            hidden_states_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let hidden_states = Tensor::from_vec(hidden_floats, (batch_size, seq_len, hidden_size), &self.device)?;

        let labels_t = Tensor::from_vec(labels.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &self.device)?;
        let mask_t = Tensor::from_vec(mask.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &self.device)?;

        // 5. Train the LoRA adapter.
        let mut optimizer = ManualAdamW::new(self.learning_rate);
        let mut last_loss = 0.0;
        for _ in 0..self.local_steps {
            last_loss = head.train_step(&hidden_states, &labels_t, &mask_t, &mut optimizer)?;
        }

        // 6. Save the adapter, encrypt it, and submit the update.
        let delta = head.save_adapter()?;
        let update = self.build_lora_update(model_id, base_checkpoint, &delta)?;
        let _new_checkpoint = self.client.submit_lora_update(update).await?;

        Ok((last_loss, delta))
    }

    fn load_artifact(&self, artifact: &TrainingArtifact) -> Result<(DnaBert2Config, DnaTokenizer, HashMap<String, Tensor>)> {
        let config = DnaBert2Config::from_bytes(&artifact.config)?;
        let tokenizer = DnaTokenizer::from_bytes(&artifact.tokenizer)?;

        let artifact_bytes = if artifact.encrypted {
            // Decrypt the artifact with the session key derived from ECDH.
            let ephemeral_public =
                PublicKey::from_slice(&artifact.ephemeral_public_key).map_err(|e| anyhow!("Invalid ephemeral public key: {}", e))?;
            let shared_secret = session::miner_shared_secret(&self.miner_secret, &ephemeral_public);
            let session_key = session::derive_session_key(&shared_secret, &artifact.session_nonce)?;
            session_key.decrypt(&artifact.artifact)?
        } else {
            artifact.artifact.clone()
        };

        // Verify the artifact signature before loading.
        let verifier =
            ArtifactVerifier::from_public_key(&artifact.auth_public_key).map_err(|e| anyhow!("Invalid auth public key: {}", e))?;
        verifier
            .verify(&artifact.artifact_hash, Some(&artifact.base_hash), &artifact.signature)
            .map_err(|e| anyhow!("Artifact signature verification failed: {}", e))?;

        let device = Device::Cpu;
        let loaded = candle_core::safetensors::load_buffer(&artifact_bytes, &device)
            .map_err(|e| anyhow!("Failed to load artifact safetensors: {}", e))?;

        let mut base_weights = HashMap::new();
        for (name, tensor) in loaded.iter() {
            base_weights.insert(name.clone(), tensor.clone());
        }

        Ok((config, tokenizer, base_weights))
    }

    fn decrypt_and_verify_forward(
        &self,
        forward: &AttestedForwardResponse,
        base_checkpoint: [u8; 32],
    ) -> Result<(Vec<u8>, ForwardSession)> {
        if forward.hidden_states.is_empty() {
            bail!("AttestedForward hidden states are empty");
        }

        // Decrypt the hidden states.
        let ephemeral_public =
            PublicKey::from_slice(&forward.ephemeral_public_key).map_err(|e| anyhow!("Invalid ephemeral public key: {}", e))?;
        let shared_secret = session::miner_shared_secret(&self.miner_secret, &ephemeral_public);
        let session_key = session::derive_session_key(&shared_secret, &forward.session_nonce)?;
        let decrypted = session_key.decrypt(&forward.hidden_states)?;

        // Recompute the hidden-states hash and verify it matches.
        let recomputed_hash = blake3_hash(&decrypted);
        if recomputed_hash != forward.hidden_states_hash {
            bail!("AttestedForward hidden-states hash mismatch");
        }

        // Verify the orchestrator signature.
        let verifier =
            ArtifactVerifier::from_public_key(&forward.auth_public_key).map_err(|e| anyhow!("Invalid auth public key: {}", e))?;
        let message_hash = build_attested_message_hash(&forward.hidden_states_hash, base_checkpoint, forward.loss);
        verifier
            .verify(&message_hash, None, &forward.signature)
            .map_err(|e| anyhow!("AttestedForward signature verification failed: {}", e))?;

        let forward_session =
            ForwardSession { ephemeral_public_key: forward.ephemeral_public_key, session_nonce: forward.session_nonce };

        Ok((decrypted, forward_session))
    }
}

impl LoraOnlyTrainer {
    fn build_lora_update(&mut self, model_id: &str, base_checkpoint: [u8; 32], delta: &[u8]) -> Result<SubmitLoRAUpdate> {
        let delta_hash = blake3_hash(delta);

        let secp = Secp256k1::new();
        let message = build_lora_delta_message_hash(&delta_hash, base_checkpoint);
        let message = Message::from_digest(message);
        let signature = secp.sign_ecdsa(&message, &self.miner_secret);
        let mut signature_bytes = [0u8; 64];
        signature_bytes.copy_from_slice(&signature.serialize_compact());

        // Encrypt the LoRA delta with the same session key used for AttestedForward.
        // The orchestrator keeps the ephemeral secret keyed by its public key.
        let (encrypted_delta, ephemeral_public_key, session_nonce) = match self.last_forward_session.as_ref() {
            Some(session) => {
                let ephemeral_public = PublicKey::from_slice(&session.ephemeral_public_key)
                    .map_err(|e| anyhow!("Invalid cached ephemeral public key: {}", e))?;
                let shared_secret = session::miner_shared_secret(&self.miner_secret, &ephemeral_public);
                let session_key = session::derive_session_key(&shared_secret, &session.session_nonce)?;
                (session_key.encrypt(delta)?, session.ephemeral_public_key, session.session_nonce)
            }
            None => (delta.to_vec(), [0u8; 33], [0u8; 12]),
        };

        Ok(SubmitLoRAUpdate {
            model_id: model_id.to_string(),
            base_checkpoint,
            lora_delta: encrypted_delta,
            lora_delta_hash: delta_hash,
            gradient_commitment: delta_hash,
            participant_weight: 1.0,
            miner_address: String::new(),
            miner_public_key: self.miner_public_key,
            encrypted: self.last_forward_session.is_some(),
            ephemeral_public_key,
            session_nonce,
            signature: signature_bytes,
            auth_public_key: self.miner_public_key,
        })
    }
}

fn build_lora_delta_message_hash(delta_hash: &[u8; 32], base_checkpoint: [u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"xenom-lora-delta-v1");
    hasher.update(delta_hash);
    hasher.update(&base_checkpoint);
    out.copy_from_slice(hasher.finalize().as_bytes());
    out
}

fn build_attested_message_hash(hidden_states_hash: &[u8; 32], base_checkpoint: [u8; 32], loss: f64) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"xenom-attested-forward-v1");
    hasher.update(hidden_states_hash);
    hasher.update(&base_checkpoint);
    hasher.update(&loss.to_le_bytes());
    out.copy_from_slice(hasher.finalize().as_bytes());
    out
}

fn blake3_hash(data: &[u8]) -> [u8; 32] {
    let hash = blake3::hash(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}
