/// Raw files that make up a Hugging Face model checkpoint.
#[derive(Debug, Clone)]
pub struct RawModelFiles {
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
}

/// AES-256-GCM encrypted model files as stored on disk (and now also sent over the wire).
#[derive(Debug, Clone)]
pub struct EncryptedModelFiles {
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
}
