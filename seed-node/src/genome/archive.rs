use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

/// Magic bytes for the `.xenom` genome archive format produced by `genome-freeze`.
pub const GENOME_FILE_MAGIC: &[u8; 8] = b"XENOGEN1";

/// Size of the `.xenom` header in bytes.
pub const GENOME_FILE_HEADER_SIZE: usize = 64;

/// Default GRCh38 fragment size used by the consensus parameters.
pub const DEFAULT_FRAGMENT_SIZE: u32 = 1_048_576;

/// Parsed `.xenom` genome archive header.
#[derive(Debug, Clone, PartialEq)]
pub struct XenomHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub dataset_version: u32,
    pub total_bases: u64,
    pub total_packed_bytes: u64,
    pub merkle_root: [u8; 32],
}

impl XenomHeader {
    pub fn magic_valid(&self) -> bool {
        &self.magic == GENOME_FILE_MAGIC
    }
}

/// In-memory representation of a `.xenom` packed genome archive.
#[derive(Debug, Clone)]
pub struct GenomeArchive {
    pub header: XenomHeader,
    pub data: Vec<u8>,
    pub fragment_size: u32,
}

impl GenomeArchive {
    /// Load a `.xenom` genome archive from disk using the default fragment size.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_with_fragment_size(path, DEFAULT_FRAGMENT_SIZE)
    }

    /// Load a `.xenom` genome archive from disk with a specific fragment size.
    pub fn load_with_fragment_size<P: AsRef<Path>>(path: P, fragment_size: u32) -> Result<Self> {
        if !fragment_size.is_multiple_of(4) {
            bail!("fragment_size must be divisible by 4");
        }
        let bytes = fs::read(&path).with_context(|| format!("Failed to read genome archive {:?}", path.as_ref()))?;
        Self::from_bytes(&bytes, fragment_size)
    }

    /// Parse a `.xenom` archive from an in-memory byte slice.
    pub fn from_bytes(bytes: &[u8], fragment_size: u32) -> Result<Self> {
        if !fragment_size.is_multiple_of(4) {
            bail!("fragment_size must be divisible by 4");
        }
        if bytes.len() < GENOME_FILE_HEADER_SIZE {
            bail!("Genome archive too small for header");
        }

        let header = Self::parse_header(&bytes[..GENOME_FILE_HEADER_SIZE])?;
        if !header.magic_valid() {
            bail!("Invalid genome archive magic: expected XENOGEN1");
        }

        let data = bytes[GENOME_FILE_HEADER_SIZE..].to_vec();
        if data.len() as u64 != header.total_packed_bytes {
            bail!("Packed data size mismatch: expected {} bytes, got {}", header.total_packed_bytes, data.len());
        }

        Ok(Self { header, data, fragment_size })
    }

    /// Number of packed bytes for one full fragment.
    pub fn packed_fragment_size(&self) -> u64 {
        (self.fragment_size / 4) as u64
    }

    /// Total number of fragments in the archive.
    /// The last fragment may be shorter than `fragment_size`.
    pub fn num_fragments(&self) -> u64 {
        let packed_frag = self.packed_fragment_size();
        if packed_frag == 0 {
            0
        } else {
            self.header.total_packed_bytes / packed_frag
        }
    }

    /// Number of bases in a given fragment.
    /// This is capped by `total_bases` so that callers do not extract padding
    /// beyond the real genome length.
    pub fn fragment_base_count(&self, fragment_idx: u64) -> Result<u32> {
        if fragment_idx >= self.num_fragments() {
            bail!("Fragment index {} out of bounds ({} fragments)", fragment_idx, self.num_fragments());
        }
        let start = fragment_idx * self.fragment_size as u64;
        let remaining = self.header.total_bases.saturating_sub(start);
        Ok((self.fragment_size as u64).min(remaining) as u32)
    }

    /// Extract a DNA sequence (as an "ACGT" string) from a specific fragment.
    ///
    /// `start` and `len` are in bases. The archive is 2-bit packed:
    /// A=0b00, C=0b01, G=0b10, T=0b11, four bases per byte, MSB-first.
    pub fn extract_sequence(&self, fragment_idx: u64, start: u32, len: u32) -> Result<String> {
        let fragment_bases = self.fragment_base_count(fragment_idx)?;
        if start.checked_add(len).ok_or_else(|| anyhow!("start+len overflow"))? > fragment_bases {
            bail!("Requested range {}..{} exceeds fragment base count {}", start, start + len, fragment_bases);
        }

        let packed_frag = self.packed_fragment_size() as usize;
        let offset = (fragment_idx * self.packed_fragment_size()) as usize;
        let end = (offset + packed_frag).min(self.data.len());
        let packed = &self.data[offset..end];

        let mut seq = String::with_capacity(len as usize);
        for i in 0..len {
            let base_offset = start + i;
            let byte_idx = (base_offset / 4) as usize;
            let shift = 6 - 2 * (base_offset % 4);
            let bits = (packed[byte_idx] >> shift) & 0b11;
            seq.push(base_from_bits(bits));
        }

        Ok(seq)
    }

    /// Unpack a whole fragment into ASCII ACGT bytes.
    ///
    /// This unpacks exactly `fragment_size` bases from the packed fragment.
    /// It is used for Merkle verification and should match the behaviour of
    /// `genome-freeze`.
    pub fn unpack_fragment(&self, fragment_idx: u64) -> Result<Vec<u8>> {
        let packed_frag = self.packed_fragment_size() as usize;
        let offset = (fragment_idx * self.packed_fragment_size()) as usize;
        let end = (offset + packed_frag).min(self.data.len());
        let packed = &self.data[offset..end];

        let mut out = Vec::with_capacity(self.fragment_size as usize);
        for &byte in packed {
            for base_in_byte in 0..4 {
                let shift = 6 - 2 * base_in_byte;
                let bits = (byte >> shift) & 0b11;
                out.push(base_from_bits(bits) as u8);
            }
        }
        Ok(out)
    }

    /// Verify the archive merkle root by re-building the merkle tree over the
    /// unpacked fragments. The leaf hash includes the fragment index.
    pub fn verify_merkle(&self) -> bool {
        if self.num_fragments() == 0 {
            return self.header.merkle_root == [0u8; 32];
        }

        let mut leaves: Vec<[u8; 32]> = Vec::with_capacity(self.num_fragments() as usize);
        for idx in 0..self.num_fragments() {
            match self.unpack_fragment(idx) {
                Ok(unpacked) => leaves.push(fragment_leaf_hash(idx, &unpacked)),
                Err(_) => return false,
            }
        }

        let root = build_merkle_root(&leaves);
        root == self.header.merkle_root
    }

    fn parse_header(bytes: &[u8]) -> Result<XenomHeader> {
        if bytes.len() < GENOME_FILE_HEADER_SIZE {
            bail!("Header buffer too short");
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);

        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let dataset_version = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let total_bases = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let total_packed_bytes = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

        let mut merkle_root = [0u8; 32];
        merkle_root.copy_from_slice(&bytes[32..64]);

        Ok(XenomHeader { magic, version, dataset_version, total_bases, total_packed_bytes, merkle_root })
    }
}

/// Compute the leaf hash for a fragment: `blake3(fragment_idx_le ‖ fragment_bytes)`.
fn fragment_leaf_hash(fragment_idx: u64, fragment: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&fragment_idx.to_le_bytes());
    h.update(fragment);
    *h.finalize().as_bytes()
}

/// Build a merkle root from a list of leaf hashes, duplicating the last leaf
/// when a level has an odd number of nodes.
fn build_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
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

/// Convert 2-bit representation to a DNA base character using the XENOGEN1 mapping.
pub fn base_from_bits(bits: u8) -> char {
    match bits & 0b11 {
        0b00 => 'A',
        0b01 => 'C',
        0b10 => 'G',
        0b11 => 'T',
        _ => unreachable!(),
    }
}

/// Number of packed bytes needed to store `base_count` bases.
pub fn packed_bytes_for(base_count: u32) -> u32 {
    base_count.div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(total_bases: u64, total_packed_bytes: u64, merkle_root: [u8; 32]) -> Vec<u8> {
        let mut header = Vec::with_capacity(GENOME_FILE_HEADER_SIZE);
        header.extend_from_slice(GENOME_FILE_MAGIC);
        header.extend_from_slice(&1u32.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&total_bases.to_le_bytes());
        header.extend_from_slice(&total_packed_bytes.to_le_bytes());
        header.extend_from_slice(&merkle_root);
        assert_eq!(header.len(), GENOME_FILE_HEADER_SIZE);
        header
    }

    #[test]
    fn test_parse_and_extract_sequence() {
        // Pack "ACGT" into one byte: A=00 (shift 6), C=01 (shift 4), G=10 (shift 2), T=11 (shift 0)
        // byte = 0b00011011 = 0x1B
        let packed = vec![0x1B];
        let merkle_root = [0u8; 32];

        let mut bytes = make_header(4, 1, merkle_root);
        bytes.extend_from_slice(&packed);

        let archive = GenomeArchive::from_bytes(&bytes, 4).unwrap();
        assert!(archive.header.magic_valid());
        assert_eq!(archive.extract_sequence(0, 0, 4).unwrap(), "ACGT");
        assert_eq!(archive.extract_sequence(0, 1, 2).unwrap(), "CG");
        assert_eq!(archive.extract_sequence(0, 0, 1).unwrap(), "A");
    }

    #[test]
    fn test_merkle_verification() {
        // Two fragments of 4 bases each: "ACGT" and "TGCA"
        let frag0 = vec![b'A', b'C', b'G', b'T'];
        let frag1 = vec![b'T', b'G', b'C', b'A'];
        let leaf0 = fragment_leaf_hash(0, &frag0);
        let leaf1 = fragment_leaf_hash(1, &frag1);

        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&leaf0);
        input[32..].copy_from_slice(&leaf1);
        let root = *blake3::hash(&input).as_bytes();

        let packed = vec![0x1B, 0b11_10_01_00]; // "ACGT" then "TGCA"
        let mut bytes = make_header(8, 2, root);
        bytes.extend_from_slice(&packed);

        let archive = GenomeArchive::from_bytes(&bytes, 4).unwrap();
        assert_eq!(archive.num_fragments(), 2);
        assert!(archive.verify_merkle());
        assert_eq!(archive.extract_sequence(1, 0, 4).unwrap(), "TGCA");
    }

    #[test]
    fn test_invalid_merkle() {
        let packed = vec![0x1B];
        let bad_root = [42u8; 32];

        let mut bytes = make_header(4, 1, bad_root);
        bytes.extend_from_slice(&packed);

        let archive = GenomeArchive::from_bytes(&bytes, 4).unwrap();
        assert!(!archive.verify_merkle());
    }
}
