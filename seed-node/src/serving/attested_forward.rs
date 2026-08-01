//! Attested forward pass for LoRA-only training.
//!
//! The orchestrator runs the base encoder on the miner's batch and returns the
//! hidden states before the LM head, together with the base loss and an
//! orchestrator signature.  The hidden states are encrypted to a per-miner
//! session key derived from ECDH.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use model_crypto::artifact_sign::ArtifactSigner;
use model_crypto::session;
use rand::RngCore;
use secp256k1::PublicKey;

use crate::model::lora_artifact::load_or_create_hierarchy;
use crate::model::manager::ModelManager;
use crate::rpc::messages::{AttestedForwardRequest, AttestedForwardResponse};

/// Produce an attested forward response for a DNABERT-2 model.
///
/// This runs the base encoder on the supplied `input_ids`, computes the LM head
/// loss against `labels`/`mask`, and returns the pre-LM-head hidden states
/// encrypted to the miner's public key and signed with the model auth key.
pub async fn attested_forward(model_manager: Arc<ModelManager>, request: AttestedForwardRequest) -> Result<AttestedForwardResponse> {
    let model_id = request.model_id.clone();
    let device = Device::Cpu;

    let (checkpoint, files) = model_manager
        .get_model_checkpoint(&model_id)
        .await
        .with_context(|| format!("Failed to load model {} for attested forward", model_id))?;

    let config = xenom_miner::model::DnaBert2Config::from_bytes(&files.config)
        .with_context(|| format!("Failed to parse config for model {}", model_id))?;
    let _tokenizer = xenom_miner::tokenizer::DnaTokenizer::from_bytes(&files.tokenizer)
        .with_context(|| format!("Failed to parse tokenizer for model {}", model_id))?;

    let model = xenom_miner::dnabert2::DnaBert2ForMaskedLM::load(config.clone(), files.weights, DType::F32, &device, None)
        .with_context(|| format!("Failed to load DNABERT-2 weights for model {}", model_id))?;

    if request.base_checkpoint != checkpoint.weights_hash {
        bail!(
            "Attested forward base checkpoint mismatch: request {} != active {}",
            hex::encode(request.base_checkpoint),
            hex::encode(checkpoint.weights_hash)
        );
    }

    let batch_size = request.input_ids.len();
    if batch_size == 0 {
        bail!("AttestedForward request has no input rows");
    }
    let seq_len = request.input_ids[0].len();

    let input_ids = Tensor::from_vec(request.input_ids.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &device)?;
    let attention_mask =
        Tensor::from_vec(request.attention_mask.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &device)?;
    let labels = Tensor::from_vec(request.labels.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &device)?;
    let mask = Tensor::from_vec(request.mask.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &device)?;

    // Run the base encoder only.  These hidden states are what the miner needs
    // for LoRA training on the LM head.
    let hidden_states =
        model.encode(&input_ids, None, Some(&attention_mask)).map_err(|e| anyhow!("Base encoder forward failed: {}", e))?;

    // Compute the base model loss for attestation.  This proves the hidden
    // states are consistent with the real base + LM head.
    let logits = model.forward_from_hidden_states(&hidden_states).map_err(|e| anyhow!("LM head forward failed: {}", e))?;
    let loss = compute_mlm_loss(&logits, &labels, &mask, &device)?;
    let loss_scalar = loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;
    if !loss_scalar.is_finite() {
        bail!("Attested forward produced non-finite loss: {}", loss_scalar);
    }

    // Serialize hidden states and compute their hash.
    let hidden_states_bytes = hidden_states
        .to_dtype(DType::F32)?
        .reshape((batch_size * seq_len * config.hidden_size,))?
        .to_vec1::<f32>()?
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>();
    let hidden_states_hash = blake3_hash(&hidden_states_bytes);

    let token_count = (batch_size * seq_len) as u32;

    // Load the model key hierarchy and sign the hidden states.
    let data_dir = Path::new(model_manager.base_path()).join(sanitize_id(&model_id));
    let hierarchy = load_or_create_hierarchy(&model_id, &data_dir).context("Failed to load model key hierarchy")?;
    let auth_key = hierarchy.auth_key().context("Failed to derive auth key")?;
    let signer = ArtifactSigner::from_auth_key(&auth_key).context("Failed to create artifact signer")?;
    let auth_public_key = signer.public_key();
    let message_hash = build_message_hash(&hidden_states_hash, request.base_checkpoint, loss_scalar);
    let signature = signer.sign(&message_hash, None).context("Failed to sign attested forward")?;

    // Encrypt the hidden states with a session key derived from ECDH.
    let miner_public_key = PublicKey::from_slice(&request.miner_public_key).map_err(|e| anyhow!("Invalid miner public key: {}", e))?;
    let (ephemeral_secret, ephemeral_public_key) = session::generate_ephemeral_keypair();
    let mut session_nonce = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut session_nonce);
    let shared_secret = session::orchestrator_shared_secret(&ephemeral_secret, &miner_public_key);
    let session_key = session::derive_session_key(&shared_secret, &session_nonce)?;
    let encrypted_hidden_states = session_key.encrypt(&hidden_states_bytes)?;

    Ok(AttestedForwardResponse {
        hidden_states_hash,
        hidden_states: encrypted_hidden_states,
        loss: loss_scalar,
        token_count,
        signature: signature.signature,
        ephemeral_public_key: ephemeral_public_key.serialize(),
        session_nonce,
        auth_public_key,
    })
}

fn build_message_hash(hidden_states_hash: &[u8; 32], base_checkpoint: [u8; 32], loss: f64) -> [u8; 32] {
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

fn sanitize_id(model_id: &str) -> String {
    model_id.chars().map(|c| if c == '/' || c == '\\' || c == ':' || c == ' ' || c == '\0' { '_' } else { c }).collect()
}

fn compute_mlm_loss(logits: &Tensor, labels: &Tensor, mask: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = logits.dims();
    let (batch, seq, vocab) = (dims[0], dims[1], dims[2]);

    let logits_flat = logits.reshape((batch * seq, vocab))?;
    let labels_flat = labels.reshape((batch * seq,))?;
    let mask_flat = mask.flatten_all()?;

    let mask_vec = mask_flat.to_vec1::<u8>()?;
    let mut positions = Vec::new();
    for (i, &m) in mask_vec.iter().enumerate() {
        if m != 0 {
            positions.push(i);
        }
    }

    if positions.is_empty() {
        return Ok(Tensor::new(0.0f32, device)?);
    }

    let positions_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
    let positions_tensor = Tensor::from_vec(positions_u32.clone(), positions_u32.len(), device)?;
    let masked_logits = logits_flat.index_select(&positions_tensor, 0)?;
    let masked_labels = labels_flat.index_select(&positions_tensor, 0)?;

    let log_sm = candle_nn::ops::log_softmax(&masked_logits, 1)?;
    let nll = candle_nn::loss::cross_entropy(&log_sm, &masked_labels.to_dtype(DType::U32)?)?;
    Ok(nll)
}
