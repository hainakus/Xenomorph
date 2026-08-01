//! LoRA-only LM head training for the secure distribution protocol.
//!
//! A miner receives only:
//! - model config
//! - tokenizer
//! - the frozen LM head base weights (transform.dense, transform.layer_norm, lm_head.bias, decoder)
//! - a LoRA adapter seed
//!
//! It then trains the LoRA adapter on `transform.dense` using attested hidden
//! states from the orchestrator, and submits the encrypted LoRA delta.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{loss, VarMap};

use crate::lora::{LinearLayer, LoraConfig, LoraLinear, ModelBuilder};
use crate::model::DnaBert2Config;
use crate::trainer::ManualAdamW;

/// A minimal trainable LM head with a LoRA adapter on `transform.dense`.
///
/// All base weights are frozen tensors; only `lora_a` and `lora_b` are
/// trainable.  This lets a miner train without the full base transformer.
pub struct LoraLmHead {
    varmap: Arc<VarMap>,
    transform_dense: LoraLinear,
    transform_layer_norm: candle_nn::LayerNorm,
    decoder: candle_nn::Linear,
}

impl LoraLmHead {
    /// Build an LM head from frozen base weights and an initial LoRA adapter.
    ///
    /// `base_weights` must contain the following keys:
    /// - `lm_head.transform.dense.weight` and `.bias`
    /// - `lm_head.transform.layer_norm.weight` and `.bias`
    /// - `lm_head.bias`
    /// - `model.embeddings.word_embeddings.weight` (used as the tied decoder)
    ///
    /// If the LoRA A/B keys are present under `lm_head.transform.dense.lora_a`
    /// and `lm_head.transform.dense.lora_b` they are loaded as trainable
    /// variables; otherwise they are initialized from scratch.
    pub fn load(
        config: &DnaBert2Config,
        base_weights: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let varmap = Arc::new(VarMap::new());
        let base_weights = Arc::new(base_weights.clone());
        let builder = ModelBuilder::new(varmap, base_weights, Some(lora_config.clone()), dtype, device.clone());

        let lm_head_builder = builder.pp("lm_head").pp("transform");
        let transform_dense = match lm_head_builder.pp("dense").linear(hidden_size, hidden_size)? {
            LinearLayer::Lora(l) => l,
            LinearLayer::Standard(_) => bail!("LoRA head expects a LoRA-enabled transform.dense"),
        };
        let transform_layer_norm = lm_head_builder.pp("layer_norm").layer_norm(hidden_size, config.layer_norm_eps)?;

        // The decoder is tied to the word embedding matrix.
        let decoder_weight = builder.pp("model").pp("embeddings").pp("word_embeddings").get_base_tensor("weight")?;
        let decoder_bias = builder.pp("lm_head").get_base_tensor("bias")?;
        let decoder = candle_nn::Linear::new(decoder_weight, Some(decoder_bias));

        Ok(Self { varmap: builder.varmap(), transform_dense, transform_layer_norm, decoder })
    }

    /// Run the LM head on pre-computed hidden states.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let h = self.transform_dense.forward(hidden_states).map_err(|e| anyhow!("transform dense: {}", e))?;
        let h = candle_nn::Activation::Gelu.forward(&h).map_err(|e| anyhow!("gelu: {}", e))?;
        let h = self.transform_layer_norm.forward(&h).map_err(|e| anyhow!("layer norm: {}", e))?;
        self.decoder.forward(&h).map_err(|e| anyhow!("decoder: {}", e))
    }

    /// Compute the masked cross-entropy loss for an MLM task.
    pub fn compute_loss(&self, hidden_states: &Tensor, labels: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let logits = self.forward(hidden_states)?;
        Self::cross_entropy_masked(&logits, labels, mask, hidden_states.device())
    }

    /// Train the LoRA adapter for one step on pre-computed hidden states.
    ///
    /// Returns the scalar loss before the update.
    pub fn train_step(&self, hidden_states: &Tensor, labels: &Tensor, mask: &Tensor, optimizer: &mut ManualAdamW) -> Result<f64> {
        let loss = self.compute_loss(hidden_states, labels, mask)?;
        let loss_scalar = loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;
        if !loss_scalar.is_finite() {
            bail!("LoRA LM head loss is not finite: {}", loss_scalar);
        }

        let grads = loss.backward()?;
        let data = self.varmap.data().lock().map_err(|e| anyhow!("VarMap poisoned: {}", e))?;
        let mut named_grads = HashMap::new();
        for (name, var) in data.iter() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                if name.contains("lora_a") || name.contains("lora_b") {
                    named_grads.insert(name.clone(), grad.to_dtype(DType::F32)?);
                }
            }
        }
        drop(data);

        optimizer.step(&self.varmap, &named_grads)?;
        Ok(loss_scalar)
    }

    /// Return a map of all trainable LoRA weights (only `lora_a` / `lora_b` keys).
    pub fn trainable_weights(&self) -> Result<HashMap<String, Tensor>> {
        let data = self.varmap.data().lock().map_err(|e| anyhow!("VarMap poisoned: {}", e))?;
        Ok(data.iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect())
    }

    /// Load a trainable LoRA adapter from a safetensors buffer.
    pub fn load_adapter(&self, bytes: &[u8]) -> Result<()> {
        let loaded =
            candle_core::safetensors::load_buffer(bytes, &self.device()).map_err(|e| anyhow!("Failed to load adapter: {}", e))?;
        let data = self.varmap.data().lock().map_err(|e| anyhow!("VarMap poisoned: {}", e))?;
        for (name, var) in data.iter() {
            if let Some(tensor) = loaded.get(name) {
                let tensor = tensor.to_device(&self.device())?.to_dtype(var.as_tensor().dtype())?;
                var.set(&tensor)?;
            }
        }
        Ok(())
    }

    /// Serialize the current LoRA A/B tensors to an in-memory safetensors buffer.
    pub fn save_adapter(&self) -> Result<Vec<u8>> {
        let data = self.varmap.data().lock().map_err(|e| anyhow!("VarMap poisoned: {}", e))?;
        let tensors: Vec<(String, &Tensor)> = data
            .iter()
            .filter(|(k, _)| k.contains("lora_a") || k.contains("lora_b"))
            .map(|(k, v)| (k.clone(), v.as_tensor()))
            .collect();
        safetensors::tensor::serialize(tensors, &None).map_err(|e| anyhow!("Failed to serialize adapter: {}", e))
    }

    fn cross_entropy_masked(logits: &Tensor, labels: &Tensor, mask: &Tensor, device: &Device) -> Result<Tensor> {
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
            bail!("No masked positions");
        }

        let positions_t = Tensor::new(positions.as_slice(), device)?;
        let masked_logits = logits_flat.index_select(&positions_t, 0)?;
        let labels_vec = labels_flat.to_vec1::<u32>()?;
        let masked_labels: Vec<u32> = positions.iter().map(|&i| labels_vec[i as usize]).collect();
        let masked_labels = Tensor::new(masked_labels.as_slice(), device)?;

        let masked_logits_f32 = masked_logits.to_dtype(DType::F32)?;
        loss::cross_entropy(&masked_logits_f32, &masked_labels).map_err(|e| anyhow!("Cross-entropy failed: {}", e))
    }

    fn device(&self) -> Device {
        let data = self.varmap.data().lock().expect("VarMap poisoned");
        data.iter().next().map(|(_, v)| v.as_tensor().device().clone()).unwrap_or(Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_lm_head_forward_and_loss() {
        let device = Device::Cpu;
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

        let mut base = HashMap::new();
        base.insert(
            "lm_head.transform.dense.weight".to_string(),
            Tensor::from_vec((0..16).map(|i| i as f32 * 0.01).collect::<Vec<_>>(), (4, 4), &device).unwrap(),
        );
        base.insert("lm_head.transform.dense.bias".to_string(), Tensor::zeros(4, DType::F32, &device).unwrap());
        base.insert("lm_head.transform.layer_norm.weight".to_string(), Tensor::ones(4, DType::F32, &device).unwrap());
        base.insert("lm_head.transform.layer_norm.bias".to_string(), Tensor::zeros(4, DType::F32, &device).unwrap());
        base.insert("lm_head.bias".to_string(), Tensor::zeros(22, DType::F32, &device).unwrap());
        base.insert(
            "model.embeddings.word_embeddings.weight".to_string(),
            Tensor::from_vec((0..88).map(|i| i as f32 * 0.001).collect::<Vec<_>>(), (22, 4), &device).unwrap(),
        );

        let lora_config =
            LoraConfig { rank: 2, alpha: 4.0, dropout: 0.0, target_modules: ["dense".to_string()].iter().cloned().collect() };
        let head = LoraLmHead::load(&config, &base, &lora_config, &device, DType::F32).unwrap();

        let hidden_states = Tensor::randn(0.0f32, 1.0, (2, 3, 4), &device).unwrap();
        let logits = head.forward(&hidden_states).unwrap();
        assert_eq!(logits.dims(), &[2, 3, 22]);

        let labels = Tensor::from_vec(vec![0u32, 1, 2, 0, 1, 2], (2, 3), &device).unwrap();
        let mask = Tensor::from_vec(vec![1u8, 0, 1, 1, 0, 1], (2, 3), &device).unwrap();
        let loss = head.compute_loss(&hidden_states, &labels, &mask).unwrap();
        assert!(loss.to_vec0::<f32>().unwrap().is_finite());

        let mut optimizer = ManualAdamW::new(1e-3);
        let loss_before = head.train_step(&hidden_states, &labels, &mask, &mut optimizer).unwrap();
        let loss_after = head.train_step(&hidden_states, &labels, &mask, &mut optimizer).unwrap();
        assert!(loss_after <= loss_before * 1.01, "LoRA head training did not converge: {} -> {}", loss_before, loss_after);
    }
}
