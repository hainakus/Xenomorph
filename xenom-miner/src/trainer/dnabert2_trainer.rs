use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer};
use std::time::Instant;

use crate::data::{MlmBatch, MlmBatchGenerator};
use crate::dnabert2::DnaBert2ForMaskedLM;
use crate::model::DnaBert2Config;
use crate::rpc::messages::TrainingBatch;
use crate::tokenizer::DnaTokenizer;
use crate::trainer::{DeviceInfo, DeviceType, Trainer, TrainingResult};

/// DNABERT-2 trainer that runs one SGD/AdamW step on a masked language modelling batch.
pub struct DnaBert2Trainer {
    model: DnaBert2ForMaskedLM,
    varmap: candle_nn::VarMap,
    generator: MlmBatchGenerator,
    device: Device,
    model_id: String,
    threads: usize,
}

impl DnaBert2Trainer {
    /// Load a trainable DNABERT-2 model and build the MLM batch generator.
    pub fn new(
        config: DnaBert2Config,
        weights: Vec<u8>,
        tokenizer: DnaTokenizer,
        device: Device,
        model_id: String,
        threads: usize,
    ) -> Result<Self> {
        let (model, varmap) = DnaBert2ForMaskedLM::load_for_training(config.clone(), weights, DType::F32, &device)
            .context("Failed to load DNABERT-2 model for training")?;
        let seq_len = config.max_position_embeddings.min(512);
        let generator = MlmBatchGenerator::new(tokenizer, seq_len);
        Ok(Self { model, varmap, generator, device, model_id, threads })
    }

    fn build_tensors(&self, batch: &MlmBatch) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let input_ids = Tensor::from_vec(
            batch.input_ids.clone(),
            (batch.batch_size, batch.seq_len),
            &self.device,
        )
        .context("Failed to create input_ids tensor")?;

        let attention_mask = Tensor::from_vec(
            batch.attention_mask.clone(),
            (batch.batch_size, batch.seq_len),
            &self.device,
        )
        .context("Failed to create attention_mask tensor")?;

        let labels = Tensor::from_vec(
            batch.labels.clone(),
            (batch.batch_size, batch.seq_len),
            &self.device,
        )
        .context("Failed to create labels tensor")?;

        let mask = Tensor::from_vec(
            batch.mask.clone(),
            (batch.batch_size, batch.seq_len),
            &self.device,
        )
        .context("Failed to create mask tensor")?;

        Ok((input_ids, attention_mask, labels, mask))
    }

    fn compute_loss(&self, logits: &Tensor, labels: &Tensor, mask: &Tensor) -> Result<Tensor> {
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

        // If nothing is masked (extremely unlikely with 15% masking), return a zero loss.
        if positions.is_empty() {
            return Ok(Tensor::new(0.0f32, &self.device)?);
        }

        let positions = Tensor::new(positions.as_slice(), &self.device)?;
        let masked_logits = logits_flat.index_select(&positions, 0)?;
        let masked_labels = labels_flat.index_select(&positions, 0)?;

        loss::cross_entropy(&masked_logits, &masked_labels).context("Failed to compute cross-entropy loss")
    }

    fn gradient_commitment(&self, grads: &candle_core::backprop::GradStore) -> Result<[u8; 32]> {
        let mut hasher = blake3::Hasher::new();
        for var in self.varmap.all_vars() {
            if let Some(grad) = grads.get(&var) {
                let values = grad.flatten_all()?.to_vec1::<f32>()?;
                for value in values {
                    hasher.update(&value.to_le_bytes());
                }
            }
        }
        Ok(*hasher.finalize().as_bytes())
    }
}

impl Trainer for DnaBert2Trainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let start = Instant::now();

        let mlm_batch = self.generator.generate(batch).context("Failed to generate MLM batch")?;
        let (input_ids, attention_mask, labels, mask) = self.build_tensors(&mlm_batch)?;

        let logits_before = self
            .model
            .forward(&input_ids, None, Some(&attention_mask))
            .context("Forward pass failed")?;
        let loss_before = self.compute_loss(&logits_before, &labels, &mask)?;
        let loss_before_scalar = loss_before.to_vec0::<f32>()? as f64;

        let mut optimizer = AdamW::new_lr(self.varmap.all_vars(), batch.learning_rate as f64)
            .context("Failed to create optimizer")?;
        let grads = loss_before.backward().context("Backward pass failed")?;
        optimizer.step(&grads).context("Optimizer step failed")?;

        let logits_after = self
            .model
            .forward(&input_ids, None, Some(&attention_mask))
            .context("Forward pass after step failed")?;
        let loss_after = self.compute_loss(&logits_after, &labels, &mask)?;
        let loss_after_scalar = loss_after.to_vec0::<f32>()? as f64;

        let gradients_commitment = self.gradient_commitment(&grads)?;

        Ok(TrainingResult {
            model_id: self.model_id.clone(),
            batch_indices: batch.data_indices.clone(),
            base_checkpoint: batch.base_checkpoint,
            loss_before: loss_before_scalar,
            loss_after: loss_after_scalar,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: DeviceType::Cpu,
            name: "DNABERT-2 CPU trainer".to_string(),
            threads: self.threads,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use std::collections::HashMap;

    fn build_tiny_tokenizer() -> DnaTokenizer {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("A".to_string(), 1);
        vocab.insert("T".to_string(), 2);
        vocab.insert("C".to_string(), 3);
        vocab.insert("G".to_string(), 4);
        vocab.insert("<mask>".to_string(), 5);

        let bpe = tokenizers::models::bpe::BPE::new(vocab, vec![]);
        let mut tokenizer = tokenizers::Tokenizer::new(bpe);
        tokenizer.add_special_tokens(&[
            tokenizers::tokenizer::AddedToken::from("<mask>", true),
            tokenizers::tokenizer::AddedToken::from("<pad>", true),
        ]);

        let bytes = serde_json::to_vec(&tokenizer).unwrap();
        DnaTokenizer::from_bytes(&bytes).unwrap()
    }

    fn insert_weight(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize], device: &Device) {
        let n = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() + 0.001).collect();
        let t = Tensor::from_vec(data, shape, device).unwrap();
        map.insert(name.to_string(), t);
    }

    fn build_tiny_safetensors() -> (DnaBert2Config, Vec<u8>) {
        let device = Device::Cpu;
        let config = DnaBert2Config {
            vocab_size: 8,
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
        insert_weight(&mut tensors, "model.embeddings.word_embeddings.weight", &[config.vocab_size, config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.token_type_embeddings.weight", &[config.type_vocab_size, config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.weight", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.bias", &[config.hidden_size], &device);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.encoder.layer.{}", i);
            insert_weight(&mut tensors, &format!("{}.attention.self.query.weight", prefix), &[config.hidden_size, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.query.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.key.weight", prefix), &[config.hidden_size, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.key.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.value.weight", prefix), &[config.hidden_size, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.value.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.dense.weight", prefix), &[config.hidden_size, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.dense.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.weight", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.bias", prefix), &[config.hidden_size], &device);

            insert_weight(&mut tensors, &format!("{}.mlp.up_proj.weight", prefix), &[config.intermediate_size * 2, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.down_proj.weight", prefix), &[config.hidden_size, config.intermediate_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.down_proj.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.layer_norm.weight", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.layer_norm.bias", prefix), &[config.hidden_size], &device);
        }

        insert_weight(&mut tensors, "lm_head.transform.dense.weight", &[config.hidden_size, config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.dense.bias", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.layer_norm.weight", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.layer_norm.bias", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.bias", &[config.vocab_size], &device);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let weights = std::fs::read(&path).unwrap();
        (config, weights)
    }

    fn dummy_batch() -> TrainingBatch {
        TrainingBatch {
            batch_id: 1,
            model_id: "dnabert2".to_string(),
            base_checkpoint: [0u8; 32],
            data_indices: vec![0, 1, 2, 3],
            target_improvement: 0.01,
            learning_rate: 0.01,
        }
    }

    #[test]
    fn test_dna_bert2_trainer_runs_and_improves() {
        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let trainer = DnaBert2Trainer::new(config, weights, tokenizer, Device::Cpu, "dnabert2".to_string(), 2).unwrap();

        let result = trainer.train(&dummy_batch()).unwrap();

        assert_eq!(result.model_id, "dnabert2");
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
        // The tiny model should usually reduce the loss after one AdamW step.
        // We allow equality in the very rare case where the random seed gives no improvement.
        assert!(result.loss_after <= result.loss_before);
    }
}
