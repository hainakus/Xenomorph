//! Attested forward pass for LoRA-only training.
//!
//! The orchestrator runs the base encoder on the miner?s batch and returns the
//! hidden states before the LM head, together with the base loss and an
//! orchestrator signature.  The miner uses the signed hidden states to train a
//! LoRA adapter without ever loading the full base weights.

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::model::manager::ModelManager;
use crate::rpc::messages::{AttestedForwardRequest, AttestedForwardResponse};

/// Produce an attested forward response for a DNABERT-2 model.
///
/// This runs the base encoder on the supplied `input_ids`, computes the LM head
/// loss against `labels`/`mask`, and returns the pre-LM-head hidden states
/// together with a hash and signature.
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

    // Serialize hidden states and compute their hash.  In production this data
    // would be encrypted to the miner?s session key before signing.
    let hidden_states_bytes = hidden_states
        .to_dtype(DType::F32)?
        .reshape((batch_size * seq_len * config.hidden_size,))?
        .to_vec1::<f32>()?
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>();
    let hidden_states_hash = blake3_hash(&hidden_states_bytes);

    let token_count = (batch_size * seq_len) as u32;

    // Signature placeholder: a real implementation would sign
    // `hidden_states_hash || base_checkpoint || request.base_checkpoint` with
    // the orchestrator's model auth key.
    let signature = [0u8; 64];

    Ok(AttestedForwardResponse {
        hidden_states_hash,
        hidden_states: hidden_states_bytes,
        loss: loss_scalar,
        token_count,
        signature,
        ephemeral_public_key: [0u8; 33],
        session_nonce: [0u8; 12],
        auth_public_key: [0u8; 33],
    })
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
            positions.push(i as u32);
        }
    }
    if positions.is_empty() {
        bail!("No masked positions in attested forward batch");
    }

    let positions_t = Tensor::new(positions.as_slice(), device)?;
    let masked_logits = logits_flat.index_select(&positions_t, 0)?;
    let labels_vec = labels_flat.to_vec1::<u32>()?;
    let masked_labels: Vec<u32> = positions.iter().map(|&i| labels_vec[i as usize]).collect();
    let masked_labels = Tensor::new(masked_labels.as_slice(), device)?;

    let masked_logits_f32 = masked_logits.to_dtype(DType::F32)?;
    candle_nn::loss::cross_entropy(&masked_logits_f32, &masked_labels).map_err(|e| anyhow!("Cross-entropy failed: {}", e))
}

fn blake3_hash(data: &[u8]) -> [u8; 32] {
    let hash = blake3::hash(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}
