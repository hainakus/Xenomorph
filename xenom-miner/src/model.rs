use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde::Deserialize;

/// Configuration for `multimolecule/dnabert2` and compatible DNABERT-2 checkpoints.
///
/// Mirrors the Hugging Face `config.json` fields that are relevant for the Rust
/// implementation. Unknown or unused keys are ignored on deserialization.
#[derive(Debug, Clone, Deserialize)]
pub struct DnaBert2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_dropout: f32,
    pub attention_dropout: f32,
    pub layer_norm_eps: f64,
    pub hidden_act: String,
    pub position_embedding_type: String,
    pub alibi_starting_size: Option<usize>,
    pub tie_word_embeddings: bool,
    pub pad_token_id: u32,
    pub mask_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub num_labels: Option<usize>,
}

impl DnaBert2Config {
    /// Parse `config.json` bytes as returned by the seed-node.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).context("Failed to parse DnaBert2 config.json")
    }

    /// A tiny config useful for unit tests of the model plumbing.
    pub fn test_config() -> Self {
        Self {
            vocab_size: 4096,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            hidden_dropout: 0.1,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".to_string(),
            position_embedding_type: "alibi".to_string(),
            alibi_starting_size: Some(512),
            tie_word_embeddings: true,
            pad_token_id: 0,
            mask_token_id: 4,
            bos_token_id: 1,
            eos_token_id: 2,
            num_labels: Some(1),
        }
    }
}

/// DNABERT-2 model container. Issue #3 only loads config and weights; the forward
/// pass will be implemented in issue #4.
pub struct DnaBert2Model {
    pub config: DnaBert2Config,
    pub varbuilder: VarBuilder<'static>,
}

impl DnaBert2Model {
    /// Load model weights from a `model.safetensors` byte buffer.
    pub fn load(config: DnaBert2Config, weights: Vec<u8>, dtype: DType, dev: &Device) -> candle_core::Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(weights, dtype, dev)?;
        Ok(Self { config, varbuilder: vb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{safetensors, Device, Tensor};
    use std::collections::HashMap;

    const SAMPLE_CONFIG: &str = r#"{
        "vocab_size": 4096,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "hidden_dropout": 0.1,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "position_embedding_type": "alibi",
        "alibi_starting_size": 512,
        "tie_word_embeddings": true,
        "pad_token_id": 0,
        "mask_token_id": 4,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "num_labels": 1
    }"#;

    #[test]
    fn test_config_from_bytes() {
        let config = DnaBert2Config::from_bytes(SAMPLE_CONFIG.as_bytes()).unwrap();
        assert_eq!(config.vocab_size, 4096);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.position_embedding_type, "alibi");
        assert_eq!(config.alibi_starting_size, Some(512));
    }

    #[test]
    fn test_model_load_from_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        // Create a tiny safetensors file with one tensor.
        let tensor = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("test.weight".to_string(), tensor);
        safetensors::save(&tensors, &path).unwrap();

        // Read it back as bytes, as the seed-node would send it.
        let weights = std::fs::read(&path).unwrap();
        let config = DnaBert2Config::test_config();

        let model = DnaBert2Model::load(config, weights, DType::F32, &Device::Cpu).unwrap();
        let loaded = model.varbuilder.get((2, 2), "test.weight").unwrap();

        let expected = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        assert_eq!(loaded.to_vec2::<f32>().unwrap(), expected.to_vec2::<f32>().unwrap());
    }
}
