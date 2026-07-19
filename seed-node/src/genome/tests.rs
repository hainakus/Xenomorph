use std::io::Write;
use std::sync::Arc;

use tempfile::NamedTempFile;

use super::archive::{base_from_bits, packed_bytes_for, GenomeArchive, HEADER_SIZE, CHUNK_INDEX_SIZE};
use super::batch_generator::{pack_sequence, GenomeBatchGenerator};
use super::storage::GenomeStorage;

fn make_test_archive(seq: &str) -> (Vec<u8>, [u8; 32]) {
    let packed = pack_sequence(seq);
    let leaf = *blake3::hash(&packed).as_bytes();
    let mut header = Vec::with_capacity(HEADER_SIZE);
    header.extend_from_slice(b"XENOM\0");
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&leaf);
    header.extend_from_slice(&1u64.to_le_bytes());

    let mut index = Vec::with_capacity(CHUNK_INDEX_SIZE);
    index.extend_from_slice(&0u64.to_le_bytes());
    index.extend_from_slice(&(seq.len() as u32).to_le_bytes());
    index.push(1u8);
    index.extend_from_slice(&0u64.to_le_bytes());

    let mut bytes = header;
    bytes.extend_from_slice(&index);
    bytes.extend_from_slice(&packed);
    (bytes, leaf)
}

#[test]
fn test_pack_unpack_consistency() {
    let seq = "ATCGATCGATCG";
    let packed = pack_sequence(seq);
    assert_eq!(packed.len(), packed_bytes_for(seq.len() as u32) as usize);

    let mut header = Vec::with_capacity(HEADER_SIZE);
    header.extend_from_slice(b"XENOM\0");
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&[0u8; 32]);
    header.extend_from_slice(&1u64.to_le_bytes());

    let mut index = Vec::with_capacity(CHUNK_INDEX_SIZE);
    index.extend_from_slice(&0u64.to_le_bytes());
    index.extend_from_slice(&(seq.len() as u32).to_le_bytes());
    index.push(1u8);
    index.extend_from_slice(&0u64.to_le_bytes());

    let mut bytes = header;
    bytes.extend_from_slice(&index);
    bytes.extend_from_slice(&packed);

    let archive = GenomeArchive::from_bytes(&bytes).unwrap();
    assert_eq!(archive.extract_sequence(0, 0, seq.len() as u32).unwrap(), seq);
}

#[test]
fn test_2bit_unpacking() {
    // Byte 0b11100100 = G(11) C(10) T(01) A(00)
    let mut header = Vec::with_capacity(HEADER_SIZE);
    header.extend_from_slice(b"XENOM\0");
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&[0u8; 32]);
    header.extend_from_slice(&1u64.to_le_bytes());

    let mut index = Vec::with_capacity(CHUNK_INDEX_SIZE);
    index.extend_from_slice(&0u64.to_le_bytes());
    index.extend_from_slice(&4u32.to_le_bytes());
    index.push(1u8);
    index.extend_from_slice(&0u64.to_le_bytes());

    let mut bytes = header;
    bytes.extend_from_slice(&index);
    bytes.extend_from_slice(&[0b11100100]);

    let archive = GenomeArchive::from_bytes(&bytes).unwrap();
    assert_eq!(archive.extract_sequence(0, 0, 4).unwrap(), "GCTA");
    assert_eq!(base_from_bits(0b00), 'A');
    assert_eq!(base_from_bits(0b01), 'T');
    assert_eq!(base_from_bits(0b10), 'C');
    assert_eq!(base_from_bits(0b11), 'G');
}

#[test]
fn test_batch_generation_determinism() {
    let (bytes, _) = make_test_archive("ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG");
    let archive = GenomeArchive::from_bytes(&bytes).unwrap();

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
    let (bytes, merkle) = make_test_archive("ATCGATCGATCGATCG");
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();

    let cache_dir = tempfile::tempdir().unwrap();
    let mut storage = GenomeStorage::new(cache_dir.path()).await.unwrap();

    let archive = storage
        .load_from_path(merkle, tmp.path())
        .await
        .unwrap();

    assert_eq!(storage.list_available(), vec![merkle]);
    assert_eq!(archive.extract_sequence(0, 0, 4).unwrap(), "ATCG");

    // Cached lookup should return the same archive without re-reading.
    let cached = storage.get_or_load(merkle, "").await.unwrap();
    assert!(Arc::ptr_eq(&archive, &cached));
}
