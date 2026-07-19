use anyhow::{Context, Result};
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
    pub type_vocab_size: usize,
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
            type_vocab_size: 2,
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

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CONFIG: &str = r#"{
        "vocab_size": 4096,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
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
        assert_eq!(config.type_vocab_size, 2);
        assert_eq!(config.position_embedding_type, "alibi");
        assert_eq!(config.alibi_starting_size, Some(512));
    }
}
