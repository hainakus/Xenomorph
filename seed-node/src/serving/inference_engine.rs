use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;
use tracing::info;

use crate::model::manager::ModelManager;

/// The kind of inference a biological model supports.
///
/// The engine dispatches `predict` to the appropriate pipeline based on this kind.
/// New models only need to be registered in `kind_for_model_id`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelKind {
    /// Conversational / instruction-tuned model (generic text generation).
    Chat,
    /// Autoregressive language model.
    CausalLM,
    /// Masked language model such as DNABERT-2.
    MaskedLM,
    /// Embedding-only model.
    Embedding,
}

/// Determine the inference kind for a Hugging Face model id.
///
/// This is the single registration point for new biological models. Add a new
/// `contains` check here and the engine will route to the correct pipeline.
fn kind_for_model_id(model_id: &str) -> ModelKind {
    let lower = model_id.to_lowercase();
    if lower.contains("dnabert") || lower.contains("nucleotide") {
        ModelKind::MaskedLM
    } else if lower.contains("hyena") || lower.contains("evo") {
        ModelKind::CausalLM
    } else {
        ModelKind::Chat
    }
}

/// A loaded biological model ready for inference.
pub struct LoadedModel {
    model: xenom_miner::dnabert2::DnaBert2ForMaskedLM,
    tokenizer: xenom_miner::tokenizer::DnaTokenizer,
    config: xenom_miner::model::DnaBert2Config,
    kind: ModelKind,
}

/// Real inference engine for biological models.
///
/// Caches loaded models in memory and exposes `predict` and `embed`.
/// `predict` automatically dispatches MaskedLM models to the MLM pipeline
/// (filling `<mask>` tokens) and would dispatch future CausalLM/Chat models to
/// their respective pipelines.
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
        let model = xenom_miner::dnabert2::DnaBert2ForMaskedLM::load(config.clone(), weights, DType::F32, &self.device)
            .with_context(|| format!("Failed to load DNABERT-2 weights for model {}", model_id))?;

        let kind = kind_for_model_id(model_id);
        info!("Loaded model for inference: {} (kind: {:?})", model_id, kind);
        Ok(Arc::new(LoadedModel { model, tokenizer, config, kind }))
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

    /// Public prediction entry point.
    pub fn predict(&self, model_id: &str, input: &str) -> Result<(String, f32)> {
        let loaded = self.get_or_load(model_id)?;
        match loaded.kind {
            ModelKind::MaskedLM => self.predict_masked_lm(&loaded, input),
            _ => Err(anyhow!(
                "Model {} is a {:?} model and the chat/prediction pipeline is not implemented for this kind",
                model_id,
                loaded.kind
            )),
        }
    }

    /// Run true Masked Language Modeling inference.
    ///
    /// * Tokenizes the prompt **without** adding special tokens, preserving the exact input
    ///   sequence length.
    /// * Finds every position whose token id equals the tokenizer's `mask_token_id`.
    /// * Runs `DnaBert2ForMaskedLM::forward` once.
    /// * Replaces each mask with the argmax-predicted token.
    /// * Decodes by concatenating raw token strings, producing a continuous DNA sequence.
    fn predict_masked_lm(&self, loaded: &LoadedModel, input: &str) -> Result<(String, f32)> {
        let mask_token = loaded.tokenizer.mask_token();

        // Normalize common aliases to the tokenizer's mask token, but never hardcode `<mask>`.
        let normalized =
            if mask_token == "<mask>" && input.contains("[MASK]") { input.replace("[MASK]", mask_token) } else { input.to_string() };

        if !normalized.contains(mask_token) {
            // No mask token in the prompt; return the input unchanged.
            return Ok((normalized, 0.0));
        }

        // Encode without special tokens so the token sequence maps 1:1 to the DNA sequence.
        let input_ids_vec = loaded.tokenizer.encode(&normalized, false)?;
        if input_ids_vec.is_empty() {
            return Err(anyhow!("Tokenizer produced no tokens for input"));
        }
        let seq_len = input_ids_vec.len();
        let mask_token_id = loaded.tokenizer.mask_token_id();

        let input_ids = Tensor::new(input_ids_vec.as_slice(), &self.device)?.reshape((1, seq_len))?;

        // Run the model and compute softmax probabilities over the vocabulary.
        let logits = loaded.model.forward(&input_ids, None, None)?;
        let probs = softmax(&logits, candle_core::D::Minus1)?;

        // Predicted token id at every position.
        let predicted_ids = logits.argmax(candle_core::D::Minus1)?; // [1, seq_len]

        // Gather the probability of each predicted token. candle's gather requires the index
        // tensor to have the same rank as the source, so expand [1, seq_len] -> [1, seq_len, 1].
        let predicted_ids_expanded = predicted_ids.unsqueeze(2)?; // [1, seq_len, 1]
        let gathered_probs = probs.gather(&predicted_ids_expanded, candle_core::D::Minus1)?; // [1, seq_len, 1]
        let gathered_probs_vec = gathered_probs
            .reshape(seq_len)?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("Failed to flatten gathered probabilities: {}", e))?;

        let predicted_ids_vec =
            predicted_ids.reshape(seq_len)?.to_vec1::<u32>().map_err(|e| anyhow!("Failed to flatten predicted ids: {}", e))?;

        // Build the output sequence, replacing mask positions with predictions.
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

        // Decode by concatenating raw token strings; this avoids the default decoder which
        // joins tokens with spaces for DNABERT-2 style BPE tokenizers.
        let output = loaded.tokenizer.decode_to_sequence(&output_ids, true)?;
        let confidence = if mask_count == 0 { 0.0 } else { total_confidence / mask_count as f32 };
        Ok((output, confidence.clamp(0.0, 1.0)))
    }

    /// Compute mean-pooled sequence embeddings for a DNA sequence.
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
        let vector =
            embeddings.reshape(embeddings.dims()[1])?.to_vec1::<f32>().map_err(|e| anyhow!("Failed to flatten embeddings: {}", e))?;
        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;
    use std::time::Instant;
    use tokenizers::models::bpe::{Vocab, BPE};
    use tokenizers::tokenizer::AddedToken;

    fn build_test_tokenizer_bytes() -> Vec<u8> {
        let mut vocab: Vocab = Vocab::new();
        for (id, token) in ["A", "T", "C", "G", "<mask>", "<pad>"].iter().enumerate() {
            vocab.insert(token.to_string(), id as u32);
        }
        let bpe = BPE::new(vocab, vec![]);
        let mut tokenizer = tokenizers::Tokenizer::new(bpe);
        tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true), AddedToken::from("<pad>", true)]);
        serde_json::to_vec(&tokenizer).expect("failed to serialize test tokenizer")
    }

    fn tiny_config(vocab_size: usize) -> xenom_miner::model::DnaBert2Config {
        xenom_miner::model::DnaBert2Config {
            vocab_size,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 32,
            max_position_embeddings: 128,
            type_vocab_size: 2,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".to_string(),
            position_embedding_type: "alibi".to_string(),
            alibi_starting_size: Some(128),
            tie_word_embeddings: true,
            pad_token_id: 5,
            mask_token_id: 4,
            bos_token_id: 1,
            eos_token_id: 2,
            num_labels: None,
        }
    }

    fn test_engine() -> InferenceEngine {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("failed to build tokio runtime");
        let manager = rt
            .block_on(ModelManager::new_with_key("/tmp/seed_node_test_models".to_string(), [0u8; 32]))
            .expect("failed to create test ModelManager");
        InferenceEngine::new(Arc::new(manager))
    }

    fn build_loaded_model() -> LoadedModel {
        let tokenizer_bytes = build_test_tokenizer_bytes();
        let tokenizer = xenom_miner::tokenizer::DnaTokenizer::from_bytes(&tokenizer_bytes).unwrap();
        let config = tiny_config(tokenizer.vocab_size);
        let device = Device::Cpu;
        let mut varmap = VarMap::new();

        // Pre-set the lm_head bias so argmax always picks a non-special nucleotide token.
        // Token 0 (A) is made slightly higher than 1-3, and 4-5 (special tokens) are strongly negative.
        let vocab_size = tokenizer.vocab_size;
        let mut bias_data = vec![-100.0f32; vocab_size];
        bias_data[0] = 110.0;
        bias_data[1] = 100.0;
        bias_data[2] = 100.0;
        bias_data[3] = 100.0;
        let bias_tensor = Tensor::new(bias_data.as_slice(), &device).unwrap();
        varmap.get(vocab_size, "lm_head.bias", candle_nn::Init::Const(0.0), DType::F32, &device).unwrap();
        varmap.set_one("lm_head.bias", &bias_tensor).unwrap();

        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = xenom_miner::dnabert2::DnaBert2ForMaskedLM::new(vb, config.clone(), &device).unwrap();
        LoadedModel { model, tokenizer, config, kind: ModelKind::MaskedLM }
    }

    #[test]
    fn test_model_kind_routing() {
        assert_eq!(kind_for_model_id("multimolecule/dnabert2"), ModelKind::MaskedLM);
        assert_eq!(kind_for_model_id("zhihan1996/DNABERT-2-117M"), ModelKind::MaskedLM);
        assert_eq!(kind_for_model_id("InstaDeepAI/nucleotide-transformer"), ModelKind::MaskedLM);
        assert_eq!(kind_for_model_id("foo/hyena-dna"), ModelKind::CausalLM);
        assert_eq!(kind_for_model_id("togethercomputer/evo-2"), ModelKind::CausalLM);
        assert_eq!(kind_for_model_id("gpt-4"), ModelKind::Chat);
    }

    #[test]
    fn test_single_mask_reconstruction() {
        let engine = test_engine();
        let loaded = build_loaded_model();
        let (output, confidence) = engine.predict_masked_lm(&loaded, "ATCG<mask>GTA").unwrap();

        assert!(!output.contains('<'), "output should not contain any special token: {}", output);
        assert!(!output.contains(' '), "output should not contain spaces: {}", output);
        assert_eq!(output.len(), 8, "output should preserve the 8 token positions: {}", output);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_multiple_mask_reconstruction() {
        let engine = test_engine();
        let loaded = build_loaded_model();
        let (output, confidence) = engine.predict_masked_lm(&loaded, "AT<mask>G<mask>TA<mask>C").unwrap();

        assert!(!output.contains('<'), "output should not contain any special token: {}", output);
        assert!(!output.contains(' '), "output should not contain spaces: {}", output);
        assert_eq!(output.len(), 9, "output should preserve the 9 token positions: {}", output);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_no_mask_returns_input() {
        let engine = test_engine();
        let loaded = build_loaded_model();
        let (output, confidence) = engine.predict_masked_lm(&loaded, "ATCGGTA").unwrap();
        assert_eq!(output, "ATCGGTA");
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_bracket_mask_alias() {
        let engine = test_engine();
        let loaded = build_loaded_model();
        let (output, _confidence) = engine.predict_masked_lm(&loaded, "ATCG[MASK]GTA").unwrap();
        assert!(!output.contains('<'), "output should not contain any special token: {}", output);
        assert!(!output.contains(' '), "output should not contain spaces: {}", output);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_predict_masked_lm_benchmark() {
        let engine = test_engine();
        let loaded = build_loaded_model();

        for mask_count in [1, 10, 100] {
            let mut input = String::new();
            for i in 1..=mask_count * 3 {
                input.push(['A', 'T', 'C', 'G'][i % 4]);
                if i % 3 == 0 {
                    input.push_str(loaded.tokenizer.mask_token());
                }
            }
            let start = Instant::now();
            let (output, _) = engine.predict_masked_lm(&loaded, &input).unwrap();
            let elapsed = start.elapsed();
            assert!(!output.contains('<'));
            assert!(!output.contains(' '));
            println!("masks={}: output_len={} elapsed={:?}", mask_count, output.len(), elapsed);
        }
    }
}
