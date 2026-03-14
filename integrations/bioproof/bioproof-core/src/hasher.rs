use crate::types::DEFAULT_CHUNK_SIZE;

/// BLAKE3 hash of `data`, returned as lowercase hex.
pub fn blake3_hex(data: &[u8]) -> String {
    hex::encode(blake3_bytes(data))
}

/// BLAKE3 hash of `data`, returned as raw 32-byte array.
pub fn blake3_bytes(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Split `data` into chunks of `chunk_size` bytes and compute BLAKE3 per chunk.
/// Returns `(chunk_hashes, file_hash)`.
pub fn chunk_and_hash(data: &[u8], chunk_size: usize) -> (Vec<[u8; 32]>, [u8; 32]) {
    let sz = if chunk_size == 0 { DEFAULT_CHUNK_SIZE } else { chunk_size };
    let file_hash  = blake3_bytes(data);
    let hashes = data.chunks(sz).map(blake3_bytes).collect();
    (hashes, file_hash)
}

/// Binary Merkle tree root over `leaves` using BLAKE3 for internal nodes.
/// Internal node hash: `blake3(left_child || right_child)`.
/// If `leaves` is empty returns `[0u8; 32]`.
pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    match leaves.len() {
        0 => [0u8; 32],
        1 => leaves[0],
        _ => {
            let mut level: Vec<[u8; 32]> = leaves.to_vec();
            while level.len() > 1 {
                let mut next = Vec::with_capacity((level.len() + 1) / 2);
                let mut i = 0;
                while i < level.len() {
                    let left  = level[i];
                    let right = *level.get(i + 1).unwrap_or(&level[i]);
                    let mut buf = [0u8; 64];
                    buf[..32].copy_from_slice(&left);
                    buf[32..].copy_from_slice(&right);
                    next.push(blake3_bytes(&buf));
                    i += 2;
                }
                level = next;
            }
            level[0]
        }
    }
}

/// Full proof pipeline for raw file bytes.
///
/// Returns `(proof_root_hex, file_hash_hex, chunk_hashes)`.
pub fn compute_proof(data: &[u8], chunk_size: usize) -> (String, String, Vec<[u8; 32]>) {
    let (chunks, file_hash) = chunk_and_hash(data, chunk_size);
    let root = merkle_root(&chunks);
    (hex::encode(root), hex::encode(file_hash), chunks)
}
