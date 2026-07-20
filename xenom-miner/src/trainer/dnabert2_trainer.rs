use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::loss;

use crate::data::{MlmBatch, MlmBatchGenerator};
use crate::dnabert2::DnaBert2ForMaskedLM;
use crate::model::DnaBert2Config;
use crate::rpc::messages::{GenomeTrainingBatchMsg, TrainingBatch};
use crate::tokenizer::DnaTokenizer;
use crate::trainer::{DeviceInfo, DeviceType, Trainer, TrainingResult};

/// AdamW learning rate cap for DNABERT-2 sized models.
/// The seed-node sends `0.01` for all trainers, but that is far too aggressive for
/// fine-tuning a pre-trained transformer on a single batch and can make the loss
/// increase after one step.
const MAX_LEARNING_RATE: f32 = 1e-5;

/// DNABERT-2 trainer that runs one SGD/AdamW step on a masked language modelling batch.
pub struct DnaBert2Trainer {
    pub(crate) model: DnaBert2ForMaskedLM,
    pub(crate) varmap: candle_nn::VarMap,
    pub(crate) generator: MlmBatchGenerator,
    pub(crate) device: Device,
    pub(crate) threads: usize,
    pub(crate) optimizer: Mutex<ManualAdamW>,
}

impl DnaBert2Trainer {
    /// Load a trainable DNABERT-2 model and build the MLM batch generator.
    pub fn new(
        config: DnaBert2Config,
        weights: Vec<u8>,
        tokenizer: DnaTokenizer,
        device: Device,
        threads: usize,
        dtype: DType,
    ) -> Result<Self> {
        let (model, varmap) = DnaBert2ForMaskedLM::load_for_training(config.clone(), weights, dtype, &device)
            .context("Failed to load DNABERT-2 model for training")?;
        let seq_len = config.max_position_embeddings.min(512);
        let generator = MlmBatchGenerator::new(tokenizer, seq_len);
        let optimizer = ManualAdamW::new(0.0);
        Ok(Self { model, varmap, generator, device, threads, optimizer: Mutex::new(optimizer) })
    }

    fn build_tensors(&self, batch: &MlmBatch) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let input_ids = Tensor::from_vec(batch.input_ids.clone(), (batch.batch_size, batch.seq_len), &self.device)
            .context("Failed to create input_ids tensor")?;

        let attention_mask = Tensor::from_vec(batch.attention_mask.clone(), (batch.batch_size, batch.seq_len), &self.device)
            .context("Failed to create attention_mask tensor")?;

        let labels = Tensor::from_vec(batch.labels.clone(), (batch.batch_size, batch.seq_len), &self.device)
            .context("Failed to create labels tensor")?;

        let mask = Tensor::from_vec(batch.mask.clone(), (batch.batch_size, batch.seq_len), &self.device)
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

    /// Compute a deterministic gradient commitment hash from a name -> tensor map.
    pub(crate) fn gradient_commitment_from_named_tensors(&self, grads: HashMap<String, Tensor>) -> Result<[u8; 32]> {
        let mut hasher = blake3::Hasher::new();
        let mut names: Vec<_> = grads.keys().cloned().collect();
        names.sort();
        for name in names {
            let grad = &grads[&name];
            // Normalize gradients to F32 for a deterministic, device-agnostic commitment.
            let grad_f32 = grad.to_dtype(DType::F32)?;
            let values = grad_f32.flatten_all()?.to_vec1::<f32>()?;
            hasher.update(name.as_bytes());
            for value in values {
                hasher.update(&value.to_le_bytes());
            }
        }
        Ok(*hasher.finalize().as_bytes())
    }

    /// Run a forward/backward pass and return the unscaled loss plus per-variable
    /// gradients moved to the CPU. `loss_scale` can be used for FP16 mixed precision.
    pub(crate) fn compute_gradients(
        &self,
        mlm_batch: &MlmBatch,
        loss_scale: f32,
    ) -> Result<(f64, HashMap<String, Tensor>)> {
        let (input_ids, attention_mask, labels, mask) = self.build_tensors(mlm_batch)?;

        let logits_before = self.model.forward(&input_ids, None, Some(&attention_mask)).context("Forward pass failed")?;
        let loss_before = self.compute_loss(&logits_before, &labels, &mask)?;
        let loss_before_scalar = loss_before.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;

        let scaled_loss = if (loss_scale - 1.0).abs() > f32::EPSILON {
            (&loss_before * (loss_scale as f64))?
        } else {
            loss_before
        };

        let grads = scaled_loss.backward().context("Backward pass failed")?;
        let named_grads = Self::grad_store_to_map(&grads, &self.varmap)?;
        Ok((loss_before_scalar, named_grads))
    }

    /// Apply named gradients to this trainer using its AdamW optimizer.
    pub(crate) fn apply_gradients(&self, named_grads: &HashMap<String, Tensor>, learning_rate: f32) -> Result<()> {
        let mut optimizer = self.optimizer.lock().map_err(|e| anyhow::anyhow!("Optimizer mutex poisoned: {}", e))?;
        let effective_lr = learning_rate.min(MAX_LEARNING_RATE);
        optimizer.set_learning_rate(effective_lr as f64);
        optimizer.step(&self.varmap, named_grads).context("Optimizer step failed")?;
        Ok(())
    }

    /// Compute the scalar loss for a batch without taking gradients.
    pub(crate) fn compute_loss_scalar(&self, mlm_batch: &MlmBatch) -> Result<f64> {
        let (input_ids, attention_mask, labels, mask) = self.build_tensors(mlm_batch)?;
        let logits = self.model.forward(&input_ids, None, Some(&attention_mask)).context("Forward pass failed")?;
        let loss = self.compute_loss(&logits, &labels, &mask)?;
        Ok(loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64)
    }

    /// Convert a `GradStore` into a `HashMap` keyed by variable name, with gradients
    /// moved to the CPU and cast to F32 for stable averaging across GPUs.
    fn grad_store_to_map(grads: &candle_core::backprop::GradStore, varmap: &candle_nn::VarMap) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        let data = varmap.data().lock().map_err(|e: PoisonError<_>| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        for (name, var) in data.iter() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let grad_cpu = grad.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                out.insert(name.clone(), grad_cpu);
            }
        }
        Ok(out)
    }
}

impl DnaBert2Trainer {
    /// Shared training step: forward, compute loss, backward, optimize, forward again.
    fn train_mlm_batch(
        &self,
        mlm_batch: &MlmBatch,
        model_id: &str,
        base_checkpoint: [u8; 32],
        batch_indices: Vec<u64>,
        learning_rate: f32,
    ) -> Result<TrainingResult> {
        let start = Instant::now();

        let (loss_before_scalar, grads) = self.compute_gradients(mlm_batch, 1.0)?;
        self.apply_gradients(&grads, learning_rate)?;
        let loss_after_scalar = self.compute_loss_scalar(mlm_batch)?;
        let gradients_commitment = self.gradient_commitment_from_named_tensors(grads)?;

        Ok(TrainingResult {
            model_id: model_id.to_string(),
            batch_indices,
            base_checkpoint,
            loss_before: loss_before_scalar,
            loss_after: loss_after_scalar,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

impl Trainer for DnaBert2Trainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let mlm_batch = self.generator.generate(batch).context("Failed to generate MLM batch")?;
        self.train_mlm_batch(&mlm_batch, &batch.model_id, batch.base_checkpoint, batch.data_indices.clone(), batch.learning_rate)
    }

    fn train_genome(&self, msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        let batch = &msg.batch;
        let mlm_batch = self
            .generator
            .generate_from_sequences(&msg.sequences, &batch.genome_merkle_root, batch.batch_id)
            .context("Failed to generate MLM batch from genome sequences")?;

        let batch_indices: Vec<u64> = batch.data_indices.iter().map(|slice| slice.chunk_idx).collect();

        self.train_mlm_batch(&mlm_batch, &batch.model_id, batch.genome_merkle_root, batch_indices, 0.01)
    }

    fn device_info(&self) -> DeviceInfo {
        let device_type = if self.device.is_cuda() {
            DeviceType::Cuda
        } else if self.device.is_metal() {
            DeviceType::Metal
        } else {
            DeviceType::Cpu
        };

        let name = match device_type {
            DeviceType::Cuda => format!("DNABERT-2 CUDA trainer ({} threads)", self.threads),
            DeviceType::Metal => format!("DNABERT-2 Metal trainer ({} threads)", self.threads),
            _ => format!("DNABERT-2 CPU trainer ({} threads)", self.threads),
        };

        DeviceInfo { device_type, name, threads: self.threads, ..Default::default() }
    }
}

/// A minimal AdamW optimizer that operates on a `VarMap` using a name-indexed
/// gradient map. This exists because `candle_core::backprop::GradStore` cannot
/// be constructed from outside the crate, which prevents feeding averaged
/// multi-GPU gradients into `candle_nn::AdamW`.
pub(crate) struct ManualAdamW {
    step_t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    /// First/second moment estimates keyed by variable name.
    moments: Mutex<HashMap<String, (Tensor, Tensor)>>,
}

impl ManualAdamW {
    pub(crate) fn new(learning_rate: f64) -> Self {
        Self {
            step_t: 0,
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            moments: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Apply named gradients (multi-GPU path where gradients are already on CPU and in F32).
    pub(crate) fn step(&mut self, varmap: &candle_nn::VarMap, named_grads: &HashMap<String, Tensor>) -> Result<()> {
        self.step_t += 1;
        let lr = self.lr;
        let lambda = self.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let scale_m = 1.0 / (1.0 - beta1.powi(self.step_t as i32));
        let scale_v = 1.0 / (1.0 - beta2.powi(self.step_t as i32));

        let mut moments = self.moments.lock().map_err(|e| anyhow::anyhow!("Moments mutex poisoned: {}", e))?;
        let data = varmap.data().lock().map_err(|e: PoisonError<_>| anyhow::anyhow!("VarMap poisoned: {}", e))?;

        for (name, var) in data.iter() {
            let grad = match named_grads.get(name) {
                Some(g) => g,
                None => continue,
            };

            let theta = var.as_tensor();
            let device = theta.device();
            let grad = grad.to_device(device)?;

            let (m, v) = moments.entry(name.clone()).or_insert_with(|| {
                // `zeros_like` cannot fail for a well-formed tensor; unwrap is safe here.
                (theta.zeros_like().unwrap(), theta.zeros_like().unwrap())
            });

            let next_m = ((&*m * beta1)? + (&grad * (1.0 - beta1))?)?;
            let next_v = ((&*v * beta2)? + (&grad.sqr()? * (1.0 - beta2))?)?;
            let m_hat = (&next_m * scale_m)?;
            let v_hat = (&next_v * scale_v)?;
            let next_theta = (theta * (1.0 - lr_lambda))?;
            let adjusted_grad = (&m_hat / (&v_hat.sqrt()? + self.eps)?)?;
            let next_theta = (&next_theta - (&adjusted_grad * lr)?)?;

            var.set(&next_theta)?;
            *m = next_m;
            *v = next_v;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use std::collections::HashMap;
    use tokenizers::models::bpe::Vocab;

    fn build_tiny_tokenizer() -> DnaTokenizer {
        let mut vocab: Vocab = Vocab::new();
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
        insert_weight(
            &mut tensors,
            "model.embeddings.token_type_embeddings.weight",
            &[config.type_vocab_size, config.hidden_size],
            &device,
        );
        insert_weight(&mut tensors, "model.embeddings.layer_norm.weight", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.bias", &[config.hidden_size], &device);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.encoder.layer.{}", i);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.query.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.query.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.key.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.key.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.value.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.value.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.output.dense.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.output.dense.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.weight", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.bias", prefix), &[config.hidden_size], &device);

            insert_weight(
                &mut tensors,
                &format!("{}.mlp.up_proj.weight", prefix),
                &[config.intermediate_size * 2, config.hidden_size],
                &device,
            );
            insert_weight(
                &mut tensors,
                &format!("{}.mlp.down_proj.weight", prefix),
                &[config.hidden_size, config.intermediate_size],
                &device,
            );
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
        let trainer = DnaBert2Trainer::new(config, weights, tokenizer, Device::Cpu, 2, DType::F32).unwrap();

        let result = trainer.train(&dummy_batch()).unwrap();

        assert_eq!(result.model_id, "dnabert2");
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
        // The tiny model should usually reduce the loss after one AdamW step.
        // We allow equality in the very rare case where the random seed gives no improvement.
        assert!(result.loss_after <= result.loss_before);
    }

    #[test]
    fn test_dna_bert2_trainer_train_genome() {
        use crate::rpc::messages::{GenomeSlice, GenomeTrainingBatch};

        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let trainer = DnaBert2Trainer::new(config, weights, tokenizer, Device::Cpu, 2, DType::F32).unwrap();

        let msg = GenomeTrainingBatchMsg {
            batch: GenomeTrainingBatch {
                batch_id: 1,
                model_id: "dnabert2".to_string(),
                genome_merkle_root: [1u8; 32],
                data_indices: vec![
                    GenomeSlice { chunk_idx: 0, start_base: 0, length: 4 },
                    GenomeSlice { chunk_idx: 1, start_base: 0, length: 4 },
                ],
                mask_ratio: 0.15,
                seq_length: 8,
            },
            sequences: vec!["ATCG".to_string(), "GCTA".to_string()],
        };

        let result = trainer.train_genome(&msg).unwrap();

        assert_eq!(result.model_id, "dnabert2");
        assert_eq!(result.batch_indices, vec![0, 1]);
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
        assert!(result.loss_after <= result.loss_before);
    }
}
