/// Raw files that make up a Hugging Face model checkpoint.
#[derive(Debug, Clone)]
pub struct RawModelFiles {
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
}
