use anyhow::{anyhow, Result};
use borsh::{BorshDeserialize, BorshSerialize};

pub struct BorshCodec;

impl BorshCodec {
    pub fn new() -> Self {
        Self
    }

    pub fn serialize<T: BorshSerialize>(&self, value: &T) -> Result<Vec<u8>> {
        borsh::to_vec(value).map_err(|e| anyhow!("Serialization failed: {}", e))
    }

    pub fn deserialize<T: BorshDeserialize>(&self, data: &[u8]) -> Result<T> {
        T::try_from_slice(data).map_err(|e| anyhow!("Deserialization failed: {}", e))
    }
}

impl Default for BorshCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, BorshSerialize, BorshDeserialize, PartialEq)]
    struct TestStruct {
        pub field1: u32,
        pub field2: String,
        pub field3: Vec<u8>,
    }

    #[test]
    fn test_codec_roundtrip() {
        let codec = BorshCodec::new();
        let original = TestStruct { field1: 42, field2: "test".to_string(), field3: vec![1, 2, 3] };

        let serialized = codec.serialize(&original).unwrap();
        let deserialized = codec.deserialize::<TestStruct>(&serialized).unwrap();

        assert_eq!(original, deserialized);
    }
}
