use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use blake3;

/// Size of the fixed `.xenom` header in bytes.
pub const HEADER_SIZE: usize = 48;

/// Size of each `ChunkIndex` entry in bytes.
pub const CHUNK_INDEX_SIZE: usize = 21;

/// Parsed `.xenom` genome archive header.
#[derive(Debug, Clone, PartialEq)]
pub struct XenomHeader {
    pub magic: [u8; 6],
    pub version: u16,
    pub merkle_root: [u8; 32],
    pub chunks: u64,
}

impl XenomHeader {
    pub fn magic_valid(&self) -> bool {
        &self.magic == b"XENOM\0"
    }
}

/// Index entry describing one packed genome chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkIndex {
    pub offset: u64,
    pub length: u32,
    pub chromosome: u8,
    pub position: u64,
}

/// In-memory representation of a `.xenom` packed genome archive.
#[derive(Debug, Clone)]
pub struct GenomeArchive {
    pub header: XenomHeader,
    pub index: Vec<ChunkIndex>,
    pub data: Vec<u8>,
}

impl GenomeArchive {
    /// Load and parse a `.xenom` genome archive from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = fs::read(&path)
            .with_context(|| format!("Failed to read genome archive {:?}", path.as_ref()))?;
        Self::from_bytes(&bytes)
    }

    /// Parse a `.xenom` archive from an in-memory byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            bail!("Genome archive too small for header");
        }

        let header = Self::parse_header(&bytes[..HEADER_SIZE])?;
        if !header.magic_valid() {
            bail!(
                "Invalid genome archive magic: expected XENOM\\0, got {:?}",
                header.magic
            );
        }

        let index_end = HEADER_SIZE as u64 + header.chunks * CHUNK_INDEX_SIZE as u64;
        if bytes.len() < index_end as usize {
            bail!("Genome archive truncated: index does not fit");
        }

        let mut index = Vec::with_capacity(header.chunks as usize);
        for i in 0..header.chunks {
            let start = HEADER_SIZE + (i as usize) * CHUNK_INDEX_SIZE;
            let end = start + CHUNK_INDEX_SIZE;
            index.push(Self::parse_chunk_index(&bytes[start..end]));
        }

        // Validate that index offsets and lengths stay inside the data section
        // before taking ownership of the payload.
        let data_len = bytes.len() - index_end as usize;
        for (i, chunk) in index.iter().enumerate() {
            let required_bytes = packed_bytes_for(chunk.length);
            let end = chunk.offset.checked_add(required_bytes as u64).ok_or_else(|| {
                anyhow!("Chunk {} offset/length overflow", i)
            })?;
            if end > data_len as u64 {
                bail!(
                    "Chunk {} extends beyond data section (offset {} + {} bytes > {})",
                    i,
                    chunk.offset,
                    required_bytes,
                    data_len
                );
            }
        }

        let data = bytes[index_end as usize..].to_vec();
        Ok(Self { header, index, data })
    }

    /// Extract a DNA sequence (as an "ATCG" string) from a specific chunk.
    ///
    /// `start` and `len` are in bases. The chunk data is 2-bit packed:
    /// A=00, T=01, C=10, G=11, four bases per byte.
    pub fn extract_sequence(&self, chunk_idx: usize, start: u32, len: u32) -> Result<String> {
        let chunk = self
            .index
            .get(chunk_idx)
            .ok_or_else(|| anyhow!("Chunk index {} out of bounds ({} chunks)", chunk_idx, self.index.len()))?;

        if start.checked_add(len).ok_or_else(|| anyhow!("start+len overflow"))? > chunk.length {
            bail!(
                "Requested range {}..{} exceeds chunk length {}",
                start,
                start + len,
                chunk.length
            );
        }

        let mut seq = String::with_capacity(len as usize);
        let start_byte = chunk.offset + (start as u64 / 4);

        for i in 0..len {
            let base_offset = start + i;
            let byte_idx = (start_byte as u64) + (base_offset as u64 / 4);
            let byte = self.data[byte_idx as usize];
            let shift = 6 - 2 * ((base_offset % 4) as u8);
            let bits = (byte >> shift) & 0b11;
            seq.push(base_from_bits(bits));
        }

        Ok(seq)
    }

    /// Verify the archive merkle root by re-building a merkle tree over the chunk data.
    pub fn verify_merkle(&self) -> bool {
        if self.header.chunks == 0 {
            return self.header.merkle_root == [0u8; 32];
        }

        let mut hashes: Vec<[u8; 32]> = self
            .index
            .iter()
            .map(|chunk| {
                let start = chunk.offset as usize;
                let end = start + packed_bytes_for(chunk.length) as usize;
                let chunk_data = &self.data[start..end];
                *blake3::hash(chunk_data).as_bytes()
            })
            .collect();

        while hashes.len() > 1 {
            let mut next = Vec::with_capacity((hashes.len() + 1) / 2);
            for pair in hashes.chunks(2) {
                let mut input = [0u8; 64];
                input[..32].copy_from_slice(&pair[0]);
                if pair.len() == 2 {
                    input[32..].copy_from_slice(&pair[1]);
                } else {
                    input[32..].copy_from_slice(&pair[0]);
                }
                next.push(*blake3::hash(&input).as_bytes());
            }
            hashes = next;
        }

        hashes[0] == self.header.merkle_root
    }

    fn parse_header(bytes: &[u8]) -> Result<XenomHeader> {
        if bytes.len() < HEADER_SIZE {
            bail!("Header buffer too short");
        }
        let mut magic = [0u8; 6];
        magic.copy_from_slice(&bytes[0..6]);

        let mut version = [0u8; 2];
        version.copy_from_slice(&bytes[6..8]);

        let mut merkle_root = [0u8; 32];
        merkle_root.copy_from_slice(&bytes[8..40]);

        let mut chunks = [0u8; 8];
        chunks.copy_from_slice(&bytes[40..48]);

        Ok(XenomHeader {
            magic,
            version: u16::from_le_bytes(version),
            merkle_root,
            chunks: u64::from_le_bytes(chunks),
        })
    }

    fn parse_chunk_index(bytes: &[u8]) -> ChunkIndex {
        let mut offset = [0u8; 8];
        offset.copy_from_slice(&bytes[0..8]);

        let mut length = [0u8; 4];
        length.copy_from_slice(&bytes[8..12]);

        let chromosome = bytes[12];

        let mut position = [0u8; 8];
        position.copy_from_slice(&bytes[13..21]);

        ChunkIndex {
            offset: u64::from_le_bytes(offset),
            length: u32::from_le_bytes(length),
            chromosome,
            position: u64::from_le_bytes(position),
        }
    }
}

/// Number of packed bytes needed to store `base_count` bases.
pub fn packed_bytes_for(base_count: u32) -> u32 {
    (base_count + 3) / 4
}

/// Convert 2-bit representation to a DNA base character.
pub fn base_from_bits(bits: u8) -> char {
    match bits & 0b11 {
        0b00 => 'A',
        0b01 => 'T',
        0b10 => 'C',
        0b11 => 'G',
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_header(chunks: u64, merkle_root: [u8; 32]) -> Vec<u8> {
        let mut header = Vec::with_capacity(HEADER_SIZE);
        header.extend_from_slice(b"XENOM\0");
        header.extend_from_slice(&1u16.to_le_bytes());
        header.extend_from_slice(&merkle_root);
        header.extend_from_slice(&chunks.to_le_bytes());
        assert_eq!(header.len(), HEADER_SIZE);
        header
    }

    fn make_chunk_index(offset: u64, length: u32, chromosome: u8, position: u64) -> Vec<u8> {
        let mut entry = Vec::with_capacity(CHUNK_INDEX_SIZE);
        entry.extend_from_slice(&offset.to_le_bytes());
        entry.extend_from_slice(&length.to_le_bytes());
        entry.push(chromosome);
        entry.extend_from_slice(&position.to_le_bytes());
        assert_eq!(entry.len(), CHUNK_INDEX_SIZE);
        entry
    }

    #[test]
    fn test_parse_and_extract_sequence() {
        // Pack "ATCG" into one byte: A=00 (shift 6), T=01 (shift 4), C=10 (shift 2), G=11 (shift 0)
        // byte = 0b00011011 = 0x1B
        let packed = vec![0x1B];
        let merkle_root = [0u8; 32];

        let mut bytes = make_test_header(1, merkle_root);
        bytes.extend_from_slice(&make_chunk_index(0, 4, 1, 1000));
        bytes.extend_from_slice(&packed);

        let archive = GenomeArchive::from_bytes(&bytes).unwrap();
        assert!(archive.header.magic_valid());
        assert_eq!(archive.index.len(), 1);
        assert_eq!(archive.extract_sequence(0, 0, 4).unwrap(), "ATCG");
        assert_eq!(archive.extract_sequence(0, 1, 2).unwrap(), "TC");
        assert_eq!(archive.extract_sequence(0, 0, 1).unwrap(), "A");
    }

    #[test]
    fn test_verify_merkle() {
        // Data with one chunk of 4 bases (one byte).
        let data = vec![0x1B];
        let leaf = *blake3::hash(&data).as_bytes();
        let mut merkle_root = [0u8; 32];
        merkle_root.copy_from_slice(&leaf);

        let mut bytes = make_test_header(1, merkle_root);
        bytes.extend_from_slice(&make_chunk_index(0, 4, 1, 1000));
        bytes.extend_from_slice(&data);

        let archive = GenomeArchive::from_bytes(&bytes).unwrap();
        assert!(archive.verify_merkle());
    }

    #[test]
    fn test_invalid_merkle() {
        let data = vec![0x1B];
        let bad_root = [42u8; 32];

        let mut bytes = make_test_header(1, bad_root);
        bytes.extend_from_slice(&make_chunk_index(0, 4, 1, 1000));
        bytes.extend_from_slice(&data);

        let archive = GenomeArchive::from_bytes(&bytes).unwrap();
        assert!(!archive.verify_merkle());
    }
}
