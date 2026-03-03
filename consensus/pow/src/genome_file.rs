//! `.xenom` genome file format — mmap-backed loader for the packed GRCh38 dataset.
//!
//! # File layout
//! ```text
//! offset  0 │  magic[8]            = b"XENOGEN1"
//! offset  8 │  version: u32        = 1  (format version)
//! offset 12 │  dataset_version: u32     (incremented on hard-fork dataset change)
//! offset 16 │  total_bases: u64         (number of nucleotide bases)
//! offset 24 │  total_packed_bytes: u64  (= ceil(total_bases / 4))
//! offset 32 │  merkle_root: [u8; 32]    (blake3 Merkle root over unpacked fragment leaves)
//!           │  ... (pad to 64 bytes)
//! offset 64 │  packed genome bytes: 2-bit encoded, 4 bases per byte, MSB-first
//!           │  A=0b00  C=0b01  G=0b10  T=0b11  N/other→A
//! ```
//!
//! # Fragment loading
//! `load_fragment(idx)` unpacks `fragment_size_bytes` bytes of ASCII ACGT from
//! the file and returns them as a `Vec<u8>`.  These are the bytes fed into
//! [`apply_mutations`](crate::genome_pow::apply_mutations) and the fitness scorer.
//!
//! # Startup Merkle verification
//! Open with `verify_merkle = true` on node startup.  The computed root must
//! match `header.merkle_root`, which in turn must match `genome_merkle_root`
//! in the consensus `Params`.  A mismatch aborts startup — the node will not
//! accept blocks with a corrupted or wrong dataset.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use kaspa_hashes::Hash;

use crate::genome_pow::{build_merkle_root, fragment_leaf_hash, GenomeDatasetLoader};

// ─── Constants ────────────────────────────────────────────────────────────────

pub const GENOME_FILE_MAGIC: &[u8; 8] = b"XENOGEN1";
pub const GENOME_FILE_FORMAT_VERSION: u32 = 1;
pub const GENOME_FILE_HEADER_SIZE: u64 = 64;

// ─── Header ───────────────────────────────────────────────────────────────────

/// Parsed representation of the 64-byte `.xenom` file header.
#[derive(Debug, Clone)]
pub struct GenomeFileHeader {
    /// Format version (currently 1).
    pub version: u32,
    /// Dataset version — incremented via hard fork when the genome dataset changes.
    pub dataset_version: u32,
    /// Number of nucleotide bases stored in the file.
    pub total_bases: u64,
    /// Number of packed bytes on disk (`ceil(total_bases / 4)`).
    pub total_packed_bytes: u64,
    /// Blake3 Merkle root computed over unpacked fragment leaves.
    pub merkle_root: [u8; 32],
}

impl GenomeFileHeader {
    /// Deserialise from the first 64 bytes of the file.
    pub fn parse(data: &[u8]) -> Result<Self, GenomeFileError> {
        if data.len() < GENOME_FILE_HEADER_SIZE as usize {
            return Err(GenomeFileError::FileTooSmall);
        }
        if &data[0..8] != GENOME_FILE_MAGIC {
            return Err(GenomeFileError::InvalidMagic);
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if version != GENOME_FILE_FORMAT_VERSION {
            return Err(GenomeFileError::UnsupportedVersion(version));
        }
        let dataset_version = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let total_bases = u64::from_le_bytes(data[16..24].try_into().unwrap());
        let total_packed_bytes = u64::from_le_bytes(data[24..32].try_into().unwrap());
        let mut merkle_root = [0u8; 32];
        merkle_root.copy_from_slice(&data[32..64]);
        Ok(Self { version, dataset_version, total_bases, total_packed_bytes, merkle_root })
    }

    /// Serialise to exactly 64 bytes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut buf = [0u8; 64];
        buf[0..8].copy_from_slice(GENOME_FILE_MAGIC);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.dataset_version.to_le_bytes());
        buf[16..24].copy_from_slice(&self.total_bases.to_le_bytes());
        buf[24..32].copy_from_slice(&self.total_packed_bytes.to_le_bytes());
        buf[32..64].copy_from_slice(&self.merkle_root);
        buf
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GenomeFileError {
    Io(std::io::Error),
    FileTooSmall,
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidFragmentSize,
    MerkleRootMismatch { expected: [u8; 32], computed: [u8; 32] },
}

impl std::fmt::Display for GenomeFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "genome file I/O error: {e}"),
            Self::FileTooSmall => write!(f, "genome file too small for 64-byte header"),
            Self::InvalidMagic => write!(f, "invalid magic bytes — expected b\"XENOGEN1\""),
            Self::UnsupportedVersion(v) => write!(f, "unsupported genome file format version {v}"),
            Self::InvalidFragmentSize => write!(f, "fragment_size_bytes must be divisible by 4"),
            Self::MerkleRootMismatch { expected, computed } => write!(
                f,
                "Merkle root mismatch — dataset corrupted or wrong file\n  expected: {}\n  computed: {}",
                hex(expected),
                hex(computed),
            ),
        }
    }
}

impl std::error::Error for GenomeFileError {}

impl From<std::io::Error> for GenomeFileError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn hex(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ─── 2-bit encoding helpers ───────────────────────────────────────────────────

/// Encode an ASCII nucleotide to its 2-bit representation.
/// A/a → 0, C/c → 1, G/g → 2, T/t → 3, N/other → 0 (A, for determinism).
#[inline]
pub fn encode_base(b: u8) -> u8 {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0,
    }
}

/// Pack a slice of ASCII nucleotide bytes into 2-bit encoding.
/// 4 bases per byte, MSB-first.  Pads with `A` if `bases.len()` is not a
/// multiple of 4.
pub fn pack_2bit(bases: &[u8]) -> Vec<u8> {
    let chunks = bases.len().div_ceil(4);
    let mut out = Vec::with_capacity(chunks);
    for chunk in bases.chunks(4) {
        let b0 = encode_base(*chunk.first().unwrap_or(&b'A'));
        let b1 = encode_base(*chunk.get(1).unwrap_or(&b'A'));
        let b2 = encode_base(*chunk.get(2).unwrap_or(&b'A'));
        let b3 = encode_base(*chunk.get(3).unwrap_or(&b'A'));
        out.push((b0 << 6) | (b1 << 4) | (b2 << 2) | b3);
    }
    out
}

/// Unpack 2-bit packed bytes to ASCII ACGT.
/// Returns exactly `packed.len() * 4` bytes.
pub fn unpack_2bit(packed: &[u8]) -> Vec<u8> {
    const TABLE: [u8; 4] = [b'A', b'C', b'G', b'T'];
    let mut out = Vec::with_capacity(packed.len() * 4);
    for &byte in packed {
        out.push(TABLE[((byte >> 6) & 0b11) as usize]);
        out.push(TABLE[((byte >> 4) & 0b11) as usize]);
        out.push(TABLE[((byte >> 2) & 0b11) as usize]);
        out.push(TABLE[(byte & 0b11) as usize]);
    }
    out
}

// ─── FileGenomeLoader ─────────────────────────────────────────────────────────

/// mmap-backed loader for `.xenom` packed genome files.
///
/// `load_fragment(idx)` returns `fragment_size_bytes` unpacked ASCII ACGT
/// bytes ready for use in [`apply_mutations`](crate::genome_pow::apply_mutations).
pub struct FileGenomeLoader {
    mmap: Mmap,
    header: GenomeFileHeader,
    /// Unpacked fragment size in bytes (bytes fed into PoW).
    fragment_size_bytes: u32,
    /// Packed fragment size on disk = `fragment_size_bytes / 4`.
    packed_frag_size: u64,
    num_fragments: u64,
}

impl FileGenomeLoader {
    /// Open a `.xenom` file.
    ///
    /// * `fragment_size_bytes` — must be divisible by 4 and match the consensus param.
    /// * `verify_merkle` — set `true` on node startup for consensus safety; `false`
    ///   for the miner (already verified at node startup).
    pub fn open(path: &Path, fragment_size_bytes: u32, verify_merkle: bool) -> Result<Self, GenomeFileError> {
        if fragment_size_bytes % 4 != 0 {
            return Err(GenomeFileError::InvalidFragmentSize);
        }
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let header = GenomeFileHeader::parse(&mmap)?;

        let packed_frag_size = (fragment_size_bytes / 4) as u64;
        let num_fragments = if packed_frag_size == 0 { 0 } else { header.total_packed_bytes / packed_frag_size };

        let loader = Self { mmap, header, fragment_size_bytes, packed_frag_size, num_fragments };

        if verify_merkle {
            let computed = loader.compute_merkle_root();
            let computed_slice: &[u8] = computed.as_ref();
            let computed_bytes: [u8; 32] = computed_slice.try_into().unwrap();
            if computed_bytes != loader.header.merkle_root {
                return Err(GenomeFileError::MerkleRootMismatch {
                    expected: loader.header.merkle_root,
                    computed: computed_bytes,
                });
            }
        }

        Ok(loader)
    }

    pub fn header(&self) -> &GenomeFileHeader {
        &self.header
    }

    pub fn num_fragments(&self) -> u64 {
        self.num_fragments
    }

    pub fn fragment_size_bytes(&self) -> u32 {
        self.fragment_size_bytes
    }

    /// Returns the file's Merkle root as a lowercase hex string, ready for
    /// comparison with `genome_merkle_root` in consensus `Params`.
    pub fn merkle_root_hex(&self) -> String {
        self.header.merkle_root.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Computes the Merkle root by hashing every fragment.
    /// Called once at startup (verify_merkle=true) — ~300 ms for 2861 fragments.
    pub fn compute_merkle_root(&self) -> Hash {
        let leaves: Vec<Hash> = (0..self.num_fragments)
            .map(|idx| {
                let unpacked = self.load_unpacked(idx);
                fragment_leaf_hash(idx, &unpacked)
            })
            .collect();
        build_merkle_root(&leaves)
    }

    /// Load the packed bytes for fragment `idx` from the mmap.
    fn load_packed_bytes(&self, idx: u64) -> &[u8] {
        let offset = (GENOME_FILE_HEADER_SIZE + idx * self.packed_frag_size) as usize;
        let end = (offset + self.packed_frag_size as usize).min(self.mmap.len());
        &self.mmap[offset..end]
    }

    fn load_unpacked(&self, idx: u64) -> Vec<u8> {
        let packed = self.load_packed_bytes(idx);
        let mut unpacked = unpack_2bit(packed);
        unpacked.truncate(self.fragment_size_bytes as usize);
        unpacked
    }
}

impl GenomeDatasetLoader for FileGenomeLoader {
    fn load_fragment(&self, idx: u64) -> Option<Vec<u8>> {
        if idx >= self.num_fragments {
            return None;
        }
        Some(self.load_unpacked(idx))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let bases = b"ACGTACGTNNACGT".to_vec();
        let packed = pack_2bit(&bases);
        let unpacked = unpack_2bit(&packed);
        // N → A by convention; pad to multiple of 4 with A
        assert_eq!(&unpacked[0..4], b"ACGT");
        assert_eq!(&unpacked[4..8], b"ACGT");
        assert_eq!(&unpacked[8..12], b"AAAC"); // NN→AA, then 'A','C'
        assert_eq!(&unpacked[12..16], b"GTAA"); // 'G','T', padded 'A','A'
    }

    #[test]
    fn encode_base_canonical() {
        assert_eq!(encode_base(b'A'), 0);
        assert_eq!(encode_base(b'a'), 0);
        assert_eq!(encode_base(b'C'), 1);
        assert_eq!(encode_base(b'G'), 2);
        assert_eq!(encode_base(b'T'), 3);
        assert_eq!(encode_base(b'N'), 0); // ambiguous → A
        assert_eq!(encode_base(b'n'), 0);
    }

    #[test]
    fn pack_density() {
        // 4 bases → 1 byte
        let packed = pack_2bit(b"ACGT");
        assert_eq!(packed.len(), 1);
        // A=00, C=01, G=10, T=11 → 0b00011011 = 0x1B
        assert_eq!(packed[0], 0b00_01_10_11);
    }

    #[test]
    fn header_roundtrip() {
        let h = GenomeFileHeader {
            version: 1,
            dataset_version: 0,
            total_bases: 3_000_000_000,
            total_packed_bytes: 750_000_000,
            merkle_root: [0xAB; 32],
        };
        let bytes = h.to_bytes();
        let parsed = GenomeFileHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.dataset_version, 0);
        assert_eq!(parsed.total_bases, 3_000_000_000);
        assert_eq!(parsed.total_packed_bytes, 750_000_000);
        assert_eq!(parsed.merkle_root, [0xAB; 32]);
    }

    #[test]
    fn header_invalid_magic() {
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(b"WRONGMAG");
        assert!(matches!(GenomeFileHeader::parse(&bytes), Err(GenomeFileError::InvalidMagic)));
    }

    #[test]
    fn header_unsupported_version() {
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(GENOME_FILE_MAGIC);
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(GenomeFileHeader::parse(&bytes), Err(GenomeFileError::UnsupportedVersion(99))));
    }
}
