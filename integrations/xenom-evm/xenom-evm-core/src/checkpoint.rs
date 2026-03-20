use revm::primitives::B256;

use crate::types::{keccak256, EvmReceipt};

pub const CHECKPOINT_VERSION_V1: u16 = 1;
pub const GENETICS_SETTLEMENT_LEAF_VERSION_V1: u16 = 1;

pub const L1_CHECKPOINT_V1_SIZE: usize = 188;
pub const GENETICS_SETTLEMENT_LEAF_V1_SIZE: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneticsSettlementLeafV1 {
    pub version: u16,
    pub block_number: u64,
    pub anchor_id: B256,
    pub payload_hash: B256,
    pub payload_len: u32,
    pub reserved: u16,
}

impl GeneticsSettlementLeafV1 {
    pub fn new(block_number: u64, anchor_id: B256, payload: &[u8]) -> Self {
        Self {
            version: GENETICS_SETTLEMENT_LEAF_VERSION_V1,
            block_number,
            anchor_id,
            payload_hash: B256::from_slice(&keccak256(payload)),
            payload_len: payload.len() as u32,
            reserved: 0,
        }
    }

    pub fn to_bytes(&self) -> [u8; GENETICS_SETTLEMENT_LEAF_V1_SIZE] {
        let mut out = [0u8; GENETICS_SETTLEMENT_LEAF_V1_SIZE];
        out[0..2].copy_from_slice(&self.version.to_be_bytes());
        out[2..10].copy_from_slice(&self.block_number.to_be_bytes());
        out[10..42].copy_from_slice(self.anchor_id.as_slice());
        out[42..74].copy_from_slice(self.payload_hash.as_slice());
        out[74..78].copy_from_slice(&self.payload_len.to_be_bytes());
        out[78..80].copy_from_slice(&self.reserved.to_be_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != GENETICS_SETTLEMENT_LEAF_V1_SIZE {
            return None;
        }

        let mut v2 = [0u8; 2];
        let mut v8 = [0u8; 8];
        let mut v4 = [0u8; 4];
        let mut a32 = [0u8; 32];
        let mut b32 = [0u8; 32];

        v2.copy_from_slice(&bytes[0..2]);
        v8.copy_from_slice(&bytes[2..10]);
        a32.copy_from_slice(&bytes[10..42]);
        b32.copy_from_slice(&bytes[42..74]);
        v4.copy_from_slice(&bytes[74..78]);
        let mut reserved = [0u8; 2];
        reserved.copy_from_slice(&bytes[78..80]);

        Some(Self {
            version: u16::from_be_bytes(v2),
            block_number: u64::from_be_bytes(v8),
            anchor_id: B256::from(a32),
            payload_hash: B256::from(b32),
            payload_len: u32::from_be_bytes(v4),
            reserved: u16::from_be_bytes(reserved),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1CheckpointV1 {
    pub version: u16,
    pub chain_id: u64,
    pub block_number: u64,
    pub timestamp_ms: u64,
    pub state_root: B256,
    pub receipts_root: B256,
    pub tx_root: B256,
    pub anchor_root: B256,
    pub genetics_settlement_root: B256,
    pub reserved: u16,
}

impl L1CheckpointV1 {
    pub fn to_bytes(&self) -> [u8; L1_CHECKPOINT_V1_SIZE] {
        let mut out = [0u8; L1_CHECKPOINT_V1_SIZE];
        out[0..2].copy_from_slice(&self.version.to_be_bytes());
        out[2..10].copy_from_slice(&self.chain_id.to_be_bytes());
        out[10..18].copy_from_slice(&self.block_number.to_be_bytes());
        out[18..26].copy_from_slice(&self.timestamp_ms.to_be_bytes());
        out[26..58].copy_from_slice(self.state_root.as_slice());
        out[58..90].copy_from_slice(self.receipts_root.as_slice());
        out[90..122].copy_from_slice(self.tx_root.as_slice());
        out[122..154].copy_from_slice(self.anchor_root.as_slice());
        out[154..186].copy_from_slice(self.genetics_settlement_root.as_slice());
        out[186..188].copy_from_slice(&self.reserved.to_be_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != L1_CHECKPOINT_V1_SIZE {
            return None;
        }

        let mut v2 = [0u8; 2];
        let mut v8 = [0u8; 8];
        let mut a32 = [0u8; 32];

        v2.copy_from_slice(&bytes[0..2]);
        let version = u16::from_be_bytes(v2);

        v8.copy_from_slice(&bytes[2..10]);
        let chain_id = u64::from_be_bytes(v8);

        v8.copy_from_slice(&bytes[10..18]);
        let block_number = u64::from_be_bytes(v8);

        v8.copy_from_slice(&bytes[18..26]);
        let timestamp_ms = u64::from_be_bytes(v8);

        a32.copy_from_slice(&bytes[26..58]);
        let state_root = B256::from(a32);

        a32.copy_from_slice(&bytes[58..90]);
        let receipts_root = B256::from(a32);

        a32.copy_from_slice(&bytes[90..122]);
        let tx_root = B256::from(a32);

        a32.copy_from_slice(&bytes[122..154]);
        let anchor_root = B256::from(a32);

        a32.copy_from_slice(&bytes[154..186]);
        let genetics_settlement_root = B256::from(a32);

        v2.copy_from_slice(&bytes[186..188]);
        let reserved = u16::from_be_bytes(v2);

        Some(Self {
            version,
            chain_id,
            block_number,
            timestamp_ms,
            state_root,
            receipts_root,
            tx_root,
            anchor_root,
            genetics_settlement_root,
            reserved,
        })
    }

    pub fn checkpoint_id(&self) -> B256 {
        B256::from_slice(&keccak256(&self.to_bytes()))
    }
}

pub fn merkle_root(leaves: &[B256]) -> B256 {
    if leaves.is_empty() {
        return B256::ZERO;
    }

    let mut layer = leaves.to_vec();
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len().div_ceil(2));
        let mut i = 0usize;
        while i < layer.len() {
            let left = layer[i];
            let right = if i + 1 < layer.len() { layer[i + 1] } else { layer[i] };
            let mut bytes = [0u8; 64];
            bytes[..32].copy_from_slice(left.as_slice());
            bytes[32..].copy_from_slice(right.as_slice());
            next.push(B256::from_slice(&keccak256(&bytes)));
            i += 2;
        }
        layer = next;
    }

    layer[0]
}

pub fn tx_root(tx_hashes: &[B256]) -> B256 {
    merkle_root(tx_hashes)
}

pub fn receipts_root(receipts: &[EvmReceipt]) -> B256 {
    let leaves: Vec<B256> = receipts
        .iter()
        .map(|r| {
            let encoded = serde_json::to_vec(r).unwrap_or_default();
            B256::from_slice(&keccak256(&encoded))
        })
        .collect();
    merkle_root(&leaves)
}

pub fn anchor_root(anchors: &[(B256, u64, Vec<u8>)]) -> B256 {
    let mut rows: Vec<(B256, u64, Vec<u8>)> = anchors.to_vec();
    rows.sort_by(|a, b| a.0.as_slice().cmp(b.0.as_slice()));

    let leaves: Vec<B256> = rows
        .iter()
        .map(|(id, block, payload)| {
            let payload_hash = keccak256(payload);
            let mut enc = Vec::with_capacity(72);
            enc.extend_from_slice(id.as_slice());
            enc.extend_from_slice(&block.to_be_bytes());
            enc.extend_from_slice(&payload_hash);
            B256::from_slice(&keccak256(&enc))
        })
        .collect();

    merkle_root(&leaves)
}

pub fn genetics_settlement_root(anchors: &[(B256, u64, Vec<u8>)]) -> B256 {
    let mut rows: Vec<(B256, u64, Vec<u8>)> = anchors
        .iter()
        .filter(|(_, _, payload)| is_genetics_settlement_payload(payload))
        .cloned()
        .collect();

    rows.sort_by(|a, b| a.0.as_slice().cmp(b.0.as_slice()));

    let leaves: Vec<B256> = rows
        .iter()
        .map(|(anchor_id, block_number, payload)| {
            let leaf = GeneticsSettlementLeafV1::new(*block_number, *anchor_id, payload);
            B256::from_slice(&keccak256(&leaf.to_bytes()))
        })
        .collect();

    merkle_root(&leaves)
}

fn is_genetics_settlement_payload(payload: &[u8]) -> bool {
    payload.windows(b"genetics-l2".len()).any(|w| w == b"genetics-l2")
        && payload.windows(b"results_root".len()).any(|w| w == b"results_root")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_binary_roundtrip_checkpoint_v1() {
        let cp = L1CheckpointV1 {
            version: CHECKPOINT_VERSION_V1,
            chain_id: 1337,
            block_number: 42,
            timestamp_ms: 1_725_000_000_000,
            state_root: B256::from([1u8; 32]),
            receipts_root: B256::from([2u8; 32]),
            tx_root: B256::from([3u8; 32]),
            anchor_root: B256::from([4u8; 32]),
            genetics_settlement_root: B256::from([5u8; 32]),
            reserved: 0,
        };

        let bytes = cp.to_bytes();
        assert_eq!(bytes.len(), L1_CHECKPOINT_V1_SIZE);

        let decoded = L1CheckpointV1::from_bytes(&bytes).expect("decode checkpoint");
        assert_eq!(decoded, cp);
        assert_ne!(decoded.checkpoint_id(), B256::ZERO);
    }

    #[test]
    fn fixed_binary_roundtrip_genetics_leaf_v1() {
        let payload = br#"{"app":"genetics-l2","results_root":"0xabc"}"#;
        let leaf = GeneticsSettlementLeafV1::new(7, B256::from([9u8; 32]), payload);

        let bytes = leaf.to_bytes();
        assert_eq!(bytes.len(), GENETICS_SETTLEMENT_LEAF_V1_SIZE);

        let decoded = GeneticsSettlementLeafV1::from_bytes(&bytes).expect("decode leaf");
        assert_eq!(decoded, leaf);
    }
}
