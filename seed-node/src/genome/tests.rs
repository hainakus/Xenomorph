use std::io::Write;
use std::sync::Arc;

use tempfile::NamedTempFile;

use super::archive::{base_from_bits, GenomeArchive, GENOME_FILE_HEADER_SIZE, GENOME_FILE_MAGIC};
use super::batch_generator::{pack_sequence, GenomeBatchGenerator};
use super::storage::GenomeStorage;

/// Build a raw `.xenom` file and return its bytes together with the merkle root.
fn make_test_archive(sequences: &[&str], fragment_size: u32) -> (Vec<u8>, [u8; 32]) {
    assert!(fragment_size % 4 == 0, "fragment size must be divisible by 4");

    let mut all_bases: Vec<u8> = sequences.iter().flat_map(|s| s.bytes()).collect();
    let total_bases = all_bases.len() as u64;

    // Pad the whole input to a multiple of `fragment_size` with 'A' so every
    // fragment is full. This keeps the test helper simple.
    let remainder = (all_bases.len() % fragment_size as usize) as u32;
    if remainder != 0 {
        let pad = fragment_size - remainder;
        all_bases.extend(std::iter::repeat(b'A').take(pad as usize));
    }

    let _packed_frag = (fragment_size / 4) as usize;
    let mut leaves: Vec<[u8; 32]> = Vec::new();
    let mut packed_data: Vec<u8> = Vec::new();

    for (idx, chunk) in all_bases.chunks(fragment_size as usize).enumerate() {
        let mut unpacked = chunk.to_vec();
        unpacked.resize(fragment_size as usize, b'A');

        let mut h = blake3::Hasher::new();
        h.update(&(idx as u64).to_le_bytes());
        h.update(&unpacked);
        leaves.push(*h.finalize().as_bytes());

        for byte_chunk in unpacked.chunks(4) {
            packed_data.push(pack_4bytes(byte_chunk));
        }
    }

    let root = build_merkle_root_from_leaves(&leaves);

    let mut header = Vec::with_capacity(GENOME_FILE_HEADER_SIZE);
    header.extend_from_slice(GENOME_FILE_MAGIC);
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
    header.extend_from_slice(&total_bases.to_le_bytes());
    header.extend_from_slice(&(packed_data.len() as u64).to_le_bytes());
    header.extend_from_slice(&root);
    assert_eq!(header.len(), GENOME_FILE_HEADER_SIZE);

    let mut bytes = header;
    bytes.extend_from_slice(&packed_data);
    (bytes, root)
}

fn pack_4bytes(chunk: &[u8]) -> u8 {
    let b0 = encode_base(*chunk.first().unwrap_or(&b'A'));
    let b1 = encode_base(*chunk.get(1).unwrap_or(&b'A'));
    let b2 = encode_base(*chunk.get(2).unwrap_or(&b'A'));
    let b3 = encode_base(*chunk.get(3).unwrap_or(&b'A'));
    (b0 << 6) | (b1 << 4) | (b2 << 2) | b3
}

fn encode_base(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0,
    }
}

fn build_merkle_root_from_leaves(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            level.push(*level.last().unwrap());
        }
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let mut input = [0u8; 64];
            input[..32].copy_from_slice(&pair[0]);
            input[32..].copy_from_slice(&pair[1]);
            next.push(*blake3::hash(&input).as_bytes());
        }
        level = next;
    }
    level[0]
}

#[test]
fn test_pack_unpack_consistency() {
    let seq = "ACGTACGTACGTACGT";
    let packed = pack_sequence(seq);
    assert_eq!(packed.len(), (seq.len() + 3) / 4);

    let (bytes, _merkle) = make_test_archive(&[seq], 16);
    let archive = GenomeArchive::from_bytes(&bytes, 16).unwrap();
    assert_eq!(archive.extract_sequence(0, 0, seq.len() as u32).unwrap(), seq);
}

#[test]
fn test_2bit_unpacking() {
    let seq = "ACGT";
    let (bytes, _merkle) = make_test_archive(&[seq], 4);
    let archive = GenomeArchive::from_bytes(&bytes, 4).unwrap();

    assert_eq!(archive.extract_sequence(0, 0, 4).unwrap(), "ACGT");
    assert_eq!(base_from_bits(0b00), 'A');
    assert_eq!(base_from_bits(0b01), 'C');
    assert_eq!(base_from_bits(0b10), 'G');
    assert_eq!(base_from_bits(0b11), 'T');
}

#[test]
fn test_batch_generation_determinism() {
    let (bytes, _merkle) = make_test_archive(&["ACGTACGTACGTACGTACGTACGTACGT"], 16);
    let archive = GenomeArchive::from_bytes(&bytes, 16).unwrap();

    let seed = [9u8; 32];
    let batch1 = GenomeBatchGenerator::new(archive.clone(), seed).generate_batch(5, 8);
    let batch2 = GenomeBatchGenerator::new(archive, seed).generate_batch(5, 8);

    assert_eq!(batch1, batch2);
    assert_eq!(batch1.data_indices.len(), 5);
    for slice in &batch1.data_indices {
        assert!(slice.length as usize <= 8);
    }
}

#[tokio::test]
async fn test_storage_load_and_verify() {
    let seq = "ACGTACGTACGTACGTACGTACGTACGTACGT";
    let (bytes, merkle) = make_test_archive(&[seq], 8);
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();

    let cache_dir = tempfile::tempdir().unwrap();
    let mut storage = GenomeStorage::new_with_fragment_size(cache_dir.path(), 8).await.unwrap();

    let archive = storage
        .load_from_path(merkle, tmp.path())
        .await
        .unwrap();

    assert_eq!(storage.list_available(), vec![merkle]);
    assert_eq!(archive.extract_sequence(0, 0, 4).unwrap(), "ACGT");

    // Cached lookup should return the same archive without re-reading.
    let cached = storage.get_or_load(merkle, "").await.unwrap();
    assert!(Arc::ptr_eq(&archive, &cached));
}
