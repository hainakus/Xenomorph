use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;
use tracing::info;

use crate::model::manager::ModelManager;

/// A loaded DNABERT-2 model ready for inference.
pub struct LoadedModel {
    model: xenom_miner::dnabert2::DnaBert2ForMaskedLM,
    tokenizer: xenom_miner::tokenizer::DnaTokenizer,
}

/// Real DNABERT-2 inference engine.
///
/// Caches loaded models in memory and exposes `predict` (masked token filling)
/// and `embed` (sequence embedding) operations.
pub struct InferenceEngine {
    model_manager: Arc<ModelManager>,
    cache: Mutex<HashMap<String, Arc<LoadedModel>>>,
    device: Device,
}

impl InferenceEngine {
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self { model_manager, cache: Mutex::new(HashMap::new()), device: Device::Cpu }
    }

    pub fn node_id(&self) -> String {
        self.model_manager.node_id().to_string()
    }

    pub fn model_manager(&self) -> Arc<ModelManager> {
        self.model_manager.clone()
    }

    fn load_sync(&self, model_id: &str) -> Result<Arc<LoadedModel>> {
        // `ModelManager::get_model_checkpoint` is async, but loading the candle model is
        // CPU-bound. Run the whole thing in the async caller's `spawn_blocking` context
        // or a dedicated runtime. Here we use `block_on` for the brief filesystem I/O.
        let runtime = tokio::runtime::Handle::try_current()?;
        let (config, tokenizer, weights) = runtime.block_on(async {
            self.model_manager.ensure_model_downloaded(model_id).await?;
            let (_checkpoint, files) = self.model_manager.get_model_checkpoint(model_id).await?;
            Ok::<_, anyhow::Error>((files.config, files.tokenizer, files.weights))
        })?;

        let config = xenom_miner::model::DnaBert2Config::from_bytes(&config)
            .with_context(|| format!("Failed to parse config for model {}", model_id))?;
        let tokenizer = xenom_miner::tokenizer::DnaTokenizer::from_bytes(&tokenizer)
            .with_context(|| format!("Failed to parse tokenizer for model {}", model_id))?;
        let model = xenom_miner::dnabert2::DnaBert2ForMaskedLM::load(config, weights, DType::F32, &self.device)
            .with_context(|| format!("Failed to load DNABERT-2 weights for model {}", model_id))?;

        info!("Loaded DNABERT-2 model for inference: {}", model_id);
        Ok(Arc::new(LoadedModel { model, tokenizer }))
    }

    fn get_or_load(&self, model_id: &str) -> Result<Arc<LoadedModel>> {
        {
            let cache = self.cache.lock().map_err(|e| anyhow!("Model cache poisoned: {}", e))?;
            if let Some(loaded) = cache.get(model_id) {
                return Ok(loaded.clone());
            }
        }

        let loaded = self.load_sync(model_id)?;
        {
            let mut cache = self.cache.lock().map_err(|e| anyhow!("Model cache poisoned: {}", e))?;
            cache.insert(model_id.to_string(), loaded.clone());
        }
        Ok(loaded)
    }

    /// Fill `<mask>` tokens in a DNA sequence and return the completed string plus
    /// the average probability of the predicted tokens.
    pub fn predict(&self, model_id: &str, input: &str) -> Result<(String, f32)> {
        let loaded = self.get_or_load(model_id)?;
        let normalized = if input.contains("<mask>") || input.contains("[MASK]") {
            input.replace("[MASK]", "<mask>")
        } else {
            // No mask token supplied: treat the last base as missing.
            let trimmed = input.trim();
            if trimmed.is_empty() {
                return Err(anyhow!("Input is empty and contains no <mask> token"));
            }
            format!("{}<mask>", trimmed)
        };

        let input_ids_vec = loaded.tokenizer.encode(&normalized, true)?;
        if input_ids_vec.is_empty() {
            return Err(anyhow!("Tokenizer produced no tokens for input"));
        }
        let seq_len = input_ids_vec.len();
        let mask_token_id = loaded.tokenizer.mask_token_id;

        let input_ids = Tensor::new(input_ids_vec.as_slice(), &self.device)?.reshape((1, seq_len))?;

        // Run the model and compute softmax probabilities over the vocabulary.
        let logits = loaded.model.forward(&input_ids, None, None)?;
        let probs = softmax(&logits, candle_core::D::Minus1)?;

        // Predicted token id at every position.
        let predicted_ids = logits.argmax(candle_core::D::Minus1)?; // [1, seq_len]

        // Gather the probability of each predicted token.
        // candle's gather requires the index tensor to have the same rank as the input,
        // so expand the 2D predicted_ids to [1, seq_len, 1] before gathering on the last dim.
        let predicted_ids_expanded = predicted_ids.unsqueeze(2)?; // [1, seq_len, 1]
        let gathered_probs = probs.gather(&predicted_ids_expanded, candle_core::D::Minus1)?; // [1, seq_len, 1]
        let gathered_probs_vec = gathered_probs
            .reshape(seq_len)?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("Failed to flatten gathered probabilities: {}", e))?;

        // Build the output sequence, replacing mask positions with predictions.
        let predicted_ids_vec = predicted_ids
            .reshape(seq_len)?
            .to_vec1::<u32>()
            .map_err(|e| anyhow!("Failed to flatten predicted ids: {}", e))?;

        let mut output_ids = input_ids_vec.clone();
        let mut mask_count = 0;
        let mut total_confidence = 0.0f32;
        for (i, &id) in input_ids_vec.iter().enumerate() {
            if id == mask_token_id {
                output_ids[i] = predicted_ids_vec[i];
                total_confidence += gathered_probs_vec[i];
                mask_count += 1;
            }
        }

        let output = loaded.tokenizer.decode(&output_ids, true)?;
        let confidence = if mask_count == 0 { 0.0 } else { total_confidence / mask_count as f32 };
        Ok((output, confidence.min(1.0).max(0.0)))
    }

    /// Compute mean-pooled DNABERT-2 embeddings for a DNA sequence.
    pub fn embed(&self, model_id: &str, input: &str) -> Result<Vec<f32>> {
        let loaded = self.get_or_load(model_id)?;
        let input = input.trim();
        if input.is_empty() {
            return Err(anyhow!("Input is empty"));
        }

        let input_ids_vec = loaded.tokenizer.encode(input, true)?;
        let seq_len = input_ids_vec.len();
        let input_ids = Tensor::new(input_ids_vec.as_slice(), &self.device)?.reshape((1, seq_len))?;

        let embeddings = loaded.model.embeddings(&input_ids)?;
        let vector = embeddings
            .reshape(embeddings.dims()[1])?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("Failed to flatten embeddings: {}", e))?;
        Ok(vector)
    }
}
