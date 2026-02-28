//! Genome PoW — Evolutionary PoW based on the Human Genome (GRCh38).
//!
//! Pipeline (per nonce attempt):
//!   1. `fragment_index(epoch_seed, nonce)`  → selects 1 MB fragment from the ~3 GB genome base
//!   2. `apply_mutations(fragment, epoch_seed)` → K deterministic mutations (Swap/Insert/Rotate/XOR/Shift)
//!   3. `compute_fitness(mutated)`           → light fitness score (entropy + GC + complexity)
//!   4. `genome_final_hash(mutated, pre_pow_hash, nonce)` → `blake3(genome ‖ header ‖ nonce)`
//!   5. Check `final_hash < target`
//!
//! Epoch update (every `epoch_len` blocks):
//!   `next_seed = blake3(median_fitness_le ‖ prev_seed)`

use kaspa_hashes::Hash;
use kaspa_math::Uint256;
use std::collections::HashSet;

/// Number of deterministic mutation steps applied to each fragment per nonce attempt.
pub const MUTATION_STEPS: usize = 8;

/// Approximate size of the human genome base (GRCh38) in bytes.
pub const GENOME_BASE_SIZE: u64 = 3_000_000_000;

// ─── Fragment selection ───────────────────────────────────────────────────────

/// Returns the fragment index for a given `(epoch_seed, nonce)` pair.
///
/// `Index = first_8_bytes_le( blake3(epoch_seed ‖ nonce_le) ) % num_fragments`
#[inline]
pub fn fragment_index(epoch_seed: &Hash, nonce: u64, fragment_size_bytes: u32) -> u64 {
    let num_fragments = GENOME_BASE_SIZE / fragment_size_bytes.max(1) as u64;
    let mut h = blake3::Hasher::new();
    h.update(epoch_seed.as_ref());
    h.update(&nonce.to_le_bytes());
    let out = h.finalize();
    let raw = u64::from_le_bytes(out.as_bytes()[0..8].try_into().unwrap());
    raw % num_fragments.max(1)
}

// ─── Deterministic mutations ──────────────────────────────────────────────────

/// Derives mutation parameters for step `step` from `blake3(epoch_seed ‖ step_le)`.
#[inline]
fn step_params(epoch_seed: &Hash, step: usize) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(epoch_seed.as_ref());
    h.update(&(step as u64).to_le_bytes());
    *h.finalize().as_bytes()
}

/// Applies one deterministic mutation to `genome` using the precomputed `params`.
///
/// Mutations (selected by `params[0] % 5`):
/// - 0 **Swap**   – exchange two byte positions
/// - 1 **Insert** – rotate a 16-byte window left and overwrite its head
/// - 2 **Rotate** – rotate a 64-byte segment by a derived amount
/// - 3 **XOR**    – XOR one byte with a derived value
/// - 4 **Shift**  – bit-shift one byte; additive perturbation at a second position
#[inline]
fn apply_step(genome: &mut [u8], params: &[u8; 32]) {
    let len = genome.len();
    if len == 0 {
        return;
    }
    let pos1 = (u32::from_le_bytes(params[1..5].try_into().unwrap()) as usize) % len;
    let pos2 = (u32::from_le_bytes(params[5..9].try_into().unwrap()) as usize) % len;
    let val = params[9];

    match params[0] % 5 {
        0 => {
            genome.swap(pos1, pos2);
        }
        1 => {
            let end = (pos1 + 16).min(len);
            genome[pos1..end].rotate_left(1);
            genome[pos1] = val;
        }
        2 => {
            let end = (pos1 + 64).min(len);
            let seg = end - pos1;
            if seg > 1 {
                genome[pos1..end].rotate_left((val as usize % seg).max(1));
            }
        }
        3 => {
            genome[pos1] ^= val;
        }
        _ => {
            genome[pos1] = if val & 1 == 0 { genome[pos1].wrapping_shl(1) } else { genome[pos1].wrapping_shr(1) };
            genome[pos2] = genome[pos2].wrapping_add(val);
        }
    }
}

/// Applies `MUTATION_STEPS` deterministic mutations to `genome` in-place.
///
/// All parameters are derived from `epoch_seed` only (not the nonce), so every
/// miner applying the same seed to the same fragment gets an identical result.
pub fn apply_mutations(genome: &mut [u8], epoch_seed: &Hash) {
    for step in 0..MUTATION_STEPS {
        let params = step_params(epoch_seed, step);
        apply_step(genome, &params);
    }
}

// ─── Fitness scoring ──────────────────────────────────────────────────────────

/// Computes the fitness score of a mutated genome fragment.
///
/// Returns a value in `[0, 3000]` as the sum of three sub-scores:
/// - **Entropy**    (0–1000): Shannon entropy of byte distribution
/// - **GC content** (0–1000): proximity to the ~50% GC optimum
/// - **Complexity** (0–1000): unique 4-gram ratio (sampled)
pub fn compute_fitness(genome: &[u8]) -> u32 {
    entropy_score(genome) + gc_content_score(genome) + cycle_complexity_score(genome)
}

/// Shannon-entropy sub-score: 0 (all bytes identical) → 1000 (uniform distribution).
fn entropy_score(data: &[u8]) -> u32 {
    if data.is_empty() {
        return 0;
    }
    let mut counts = [0u32; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let len = data.len() as f64;
    let entropy: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum();
    // Max entropy for bytes = log2(256) = 8.0 bits
    ((entropy / 8.0) * 1000.0).min(1000.0) as u32
}

/// GC-content sub-score: peaks at 50 % GC (human genome optimum).
/// Uses ASCII values: `G` = 0x47, `C` = 0x43.
fn gc_content_score(data: &[u8]) -> u32 {
    if data.is_empty() {
        return 0;
    }
    let gc = data.iter().filter(|&&b| b == b'G' || b == b'C').count();
    let ratio = gc as f64 / data.len() as f64;
    let deviation = (ratio - 0.5_f64).abs();
    ((1.0 - deviation * 2.0).max(0.0) * 1000.0) as u32
}

/// 4-gram complexity sub-score: more unique 4-byte windows → higher score.
/// Sampled over the first `SAMPLE` bytes for performance.
fn cycle_complexity_score(data: &[u8]) -> u32 {
    const SAMPLE: usize = 4096;
    if data.len() < 4 {
        return 0;
    }
    let sample = &data[..data.len().min(SAMPLE)];
    let total = sample.len() - 3;
    let mut seen: HashSet<u32> = HashSet::with_capacity(total);
    for w in sample.windows(4) {
        seen.insert(u32::from_le_bytes(w.try_into().unwrap()));
    }
    let ratio = seen.len() as f64 / total as f64;
    (ratio * 1000.0).min(1000.0) as u32
}

// ─── Final hash & epoch seed ──────────────────────────────────────────────────

/// Hashes a mutated genome fragment to a 32-byte digest.
///
/// This intermediate hash can be pre-computed once per fragment per epoch (since
/// mutations depend only on `epoch_seed`, not on the nonce), enabling efficient
/// GPU mining where each nonce only needs one small blake3 call.
#[inline]
pub fn genome_fragment_pow_hash(mutated_genome: &[u8]) -> [u8; 32] {
    *blake3::hash(mutated_genome).as_bytes()
}

/// Computes the Genome PoW value:
/// `blake3( blake3(mutated_genome) ‖ pre_pow_hash ‖ nonce_le )`.
///
/// The two-step design lets GPU miners pre-compute `blake3(mutated_genome)` for
/// every fragment once per epoch; per-nonce work is then a single 72-byte blake3.
#[inline]
pub fn genome_final_hash(genome: &[u8], pre_pow_hash: &Hash, nonce: u64) -> Uint256 {
    let genome_h = genome_fragment_pow_hash(genome);
    let mut h = blake3::Hasher::new();
    h.update(&genome_h);
    h.update(pre_pow_hash.as_ref());
    h.update(&nonce.to_le_bytes());
    Uint256::from_le_bytes(*h.finalize().as_bytes())
}

/// Computes the next epoch seed: `blake3(epoch_score_le ‖ prev_seed)`.
///
/// Called once every `epoch_len` blocks using the median fitness of the window.
pub fn next_epoch_seed(epoch_score: u32, prev_seed: &Hash) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(&epoch_score.to_le_bytes());
    h.update(prev_seed.as_ref());
    Hash::from_bytes(*h.finalize().as_bytes())
}

// ─── GenomePowState ───────────────────────────────────────────────────────────

/// Pre-computed state for mining / validating a single block using Genome PoW.
///
/// The caller is responsible for supplying the correct raw fragment bytes
/// (loaded from the local genome dataset and identified by
/// `fragment_index(epoch_seed, nonce, fragment_size_bytes)`).
pub struct GenomePowState {
    pub pre_pow_hash: Hash,
    pub target: Uint256,
    pub epoch_seed: Hash,
    pub fragment_size_bytes: u32,
}

impl GenomePowState {
    pub fn new(pre_pow_hash: Hash, target: Uint256, epoch_seed: Hash, fragment_size_bytes: u32) -> Self {
        Self { pre_pow_hash, target, epoch_seed, fragment_size_bytes }
    }

    /// Returns the fragment index for `nonce`.
    #[inline]
    pub fn fragment_index_for(&self, nonce: u64) -> u64 {
        fragment_index(&self.epoch_seed, nonce, self.fragment_size_bytes)
    }

    /// Validates `nonce` against the provided raw genome `fragment`.
    ///
    /// Returns `(valid, pow_value, fitness_score)`.
    #[inline]
    pub fn check_pow_with_fragment(&self, nonce: u64, fragment: &[u8]) -> (bool, Uint256, u32) {
        let mut genome = fragment.to_vec();
        apply_mutations(&mut genome, &self.epoch_seed);
        let fitness = compute_fitness(&genome);
        let pow = genome_final_hash(&genome, &self.pre_pow_hash, nonce);
        (pow <= self.target, pow, fitness)
    }
}

// ─── Merkle proof verification ────────────────────────────────────────────────

/// Leaf hash for a genome fragment: `blake3(fragment_idx_le ‖ fragment_bytes)`.
pub fn fragment_leaf_hash(fragment_idx: u64, fragment: &[u8]) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(&fragment_idx.to_le_bytes());
    h.update(fragment);
    Hash::from_bytes(*h.finalize().as_bytes())
}

/// Internal node hash: `blake3(left ‖ right)`.
#[inline]
fn merkle_node_hash(left: &Hash, right: &Hash) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(left.as_ref());
    h.update(right.as_ref());
    Hash::from_bytes(*h.finalize().as_bytes())
}

/// Inclusion proof for a single genome fragment in the Merkle tree.
///
/// `siblings[0]` is the sibling at the leaf level; `siblings[last]` is the
/// sibling just below the root.  `leaf_index` is the 0-based fragment index
/// within the tree (equals the fragment index into the dataset).
#[derive(Debug, Clone)]
pub struct GenomeMerkleProof {
    pub leaf_index: u64,
    pub siblings: Vec<Hash>,
}

impl GenomeMerkleProof {
    /// Verifies this proof against `expected_root_hex` (the `genome_merkle_root`
    /// parameter, stored as a lowercase hex string).
    ///
    /// Returns `true` iff the fragment authenticates correctly.
    pub fn verify(&self, expected_root_hex: &str, fragment_idx: u64, fragment: &[u8]) -> bool {
        let expected = match parse_hash_hex(expected_root_hex) {
            Some(h) => h,
            None => return false,
        };
        let mut current = fragment_leaf_hash(fragment_idx, fragment);
        let mut index = self.leaf_index;
        for sibling in &self.siblings {
            current = if index & 1 == 0 {
                merkle_node_hash(&current, sibling)
            } else {
                merkle_node_hash(sibling, &current)
            };
            index >>= 1;
        }
        current == expected
    }
}

/// Parses a lowercase hex string (64 chars) into a `Hash`.
fn parse_hash_hex(hex: &str) -> Option<Hash> {
    if hex.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        bytes[i] = (hi << 4) | lo;
    }
    Some(Hash::from_bytes(bytes))
}

#[inline]
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Builds the Merkle root over `num_fragments` leaf hashes where each leaf is
/// `fragment_leaf_hash(idx, fragment_data_for(idx))`.
///
/// Used by the genome dataset builder / test utilities.  For the full genome
/// the leaves are computed from the actual GRCh38 fragments; here they are
/// synthesised for testing purposes.
pub fn build_merkle_root(leaves: &[Hash]) -> Hash {
    if leaves.is_empty() {
        return Hash::from_bytes([0u8; 32]);
    }
    let mut level: Vec<Hash> = leaves.to_vec();
    // Pad to even length by duplicating the last leaf
    while level.len() > 1 {
        if level.len() % 2 == 1 {
            let last = *level.last().unwrap();
            level.push(last);
        }
        let next: Vec<Hash> = level.chunks(2).map(|pair| merkle_node_hash(&pair[0], &pair[1])).collect();
        level = next;
    }
    level[0]
}

// ─── Fragment cache ───────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Thread-safe LRU-like cache for loaded genome fragments.
///
/// Entries are evicted by insertion order once `max_entries` is exceeded.
/// This avoids pulling in an external LRU crate while still bounding memory.
pub struct GenomeFragmentCache {
    inner: Mutex<FragmentCacheInner>,
}

struct FragmentCacheInner {
    map: HashMap<u64, Arc<Vec<u8>>>,
    order: std::collections::VecDeque<u64>,
    max_entries: usize,
}

impl GenomeFragmentCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            inner: Mutex::new(FragmentCacheInner {
                map: HashMap::new(),
                order: std::collections::VecDeque::new(),
                max_entries: max_entries.max(1),
            }),
        }
    }

    pub fn get(&self, idx: u64) -> Option<Arc<Vec<u8>>> {
        self.inner.lock().unwrap().map.get(&idx).cloned()
    }

    pub fn insert(&self, idx: u64, fragment: Arc<Vec<u8>>) {
        let mut inner = self.inner.lock().unwrap();
        if inner.map.contains_key(&idx) {
            return;
        }
        while inner.map.len() >= inner.max_entries {
            if let Some(oldest) = inner.order.pop_front() {
                inner.map.remove(&oldest);
            }
        }
        inner.map.insert(idx, fragment);
        inner.order.push_back(idx);
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─── GenomeDatasetLoader ──────────────────────────────────────────────────────

/// Abstraction over the source of genome fragment data.
///
/// The production implementation reads 1 MB chunks from the local GRCh38
/// flat file.  Tests / devnet supply a `SyntheticLoader` that generates
/// fragments deterministically from the fragment index.
pub trait GenomeDatasetLoader: Send + Sync {
    /// Returns the raw bytes for fragment `idx`, or `None` if unavailable.
    fn load_fragment(&self, idx: u64) -> Option<Vec<u8>>;
}

/// Loader that synthesises fragments deterministically from the fragment index
/// and an epoch seed.  Used during development and when the real genome dataset
/// is not present on disk.
pub struct SyntheticLoader {
    fragment_size_bytes: u32,
    epoch_seed: Hash,
}

impl SyntheticLoader {
    pub fn new(fragment_size_bytes: u32, epoch_seed: Hash) -> Self {
        Self { fragment_size_bytes, epoch_seed }
    }
}

impl GenomeDatasetLoader for SyntheticLoader {
    fn load_fragment(&self, idx: u64) -> Option<Vec<u8>> {
        let size = self.fragment_size_bytes as usize;
        let mut out = Vec::with_capacity(size);
        let mut chunk = 0u64;
        while out.len() < size {
            let mut h = blake3::Hasher::new();
            h.update(&idx.to_le_bytes());
            h.update(self.epoch_seed.as_ref());
            h.update(&chunk.to_le_bytes());
            out.extend_from_slice(h.finalize().as_bytes());
            chunk += 1;
        }
        out.truncate(size);
        Some(out)
    }
}

/// Cached wrapper around any `GenomeDatasetLoader`.
pub struct CachedLoader<L: GenomeDatasetLoader> {
    inner: L,
    cache: GenomeFragmentCache,
}

impl<L: GenomeDatasetLoader> CachedLoader<L> {
    pub fn new(inner: L, cache_entries: usize) -> Self {
        Self { inner, cache: GenomeFragmentCache::new(cache_entries) }
    }
}

impl<L: GenomeDatasetLoader> GenomeDatasetLoader for CachedLoader<L> {
    fn load_fragment(&self, idx: u64) -> Option<Vec<u8>> {
        if let Some(cached) = self.cache.get(idx) {
            return Some(cached.as_ref().clone());
        }
        let fragment = self.inner.load_fragment(idx)?;
        self.cache.insert(idx, Arc::new(fragment.clone()));
        Some(fragment)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(byte: u8) -> Hash {
        Hash::from_bytes([byte; 32])
    }

    #[test]
    fn fragment_index_is_deterministic() {
        let s = seed(1);
        assert_eq!(fragment_index(&s, 42, 1024 * 1024), fragment_index(&s, 42, 1024 * 1024));
    }

    #[test]
    fn fragment_index_changes_with_nonce() {
        let s = seed(1);
        let fs = 1024 * 1024u32;
        let i0 = fragment_index(&s, 0, fs);
        let i1 = fragment_index(&s, 1, fs);
        let i2 = fragment_index(&s, 2, fs);
        // With 2861 possible fragments, all three should almost certainly differ
        assert!(i0 != i1 || i1 != i2, "consecutive nonces produced identical fragment indices");
    }

    #[test]
    fn mutations_are_deterministic() {
        let s = seed(0xab);
        let original: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        let mut g1 = original.clone();
        let mut g2 = original.clone();
        apply_mutations(&mut g1, &s);
        apply_mutations(&mut g2, &s);
        assert_eq!(g1, g2);
        assert_ne!(g1, original);
    }

    #[test]
    fn mutations_change_with_seed() {
        let original: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        let mut g1 = original.clone();
        let mut g2 = original.clone();
        apply_mutations(&mut g1, &seed(0x01));
        apply_mutations(&mut g2, &seed(0x02));
        assert_ne!(g1, g2);
    }

    #[test]
    fn fitness_in_range() {
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let score = compute_fitness(&data);
        assert!(score <= 3000, "fitness={score} exceeds max 3000");
    }

    #[test]
    fn next_epoch_seed_deterministic() {
        let s = seed(7);
        assert_eq!(next_epoch_seed(500, &s), next_epoch_seed(500, &s));
        assert_ne!(next_epoch_seed(500, &s), next_epoch_seed(501, &s));
    }

    #[test]
    fn genome_pow_state_round_trip() {
        let pre_pow = Hash::from_bytes([0xde; 32]);
        let target = Uint256::MAX;
        let epoch_seed = seed(0x11);
        let state = GenomePowState::new(pre_pow, target, epoch_seed, 1024 * 1024);

        let fragment = vec![b'A'; 256];
        let nonce = 99u64;
        let (valid, pow, fitness) = state.check_pow_with_fragment(nonce, &fragment);
        assert!(valid, "all-A fragment with MAX target must be valid, pow={pow:?}");
        assert!(fitness <= 3000);
    }

    // ── Merkle proof tests ───────────────────────────────────────────────────

    fn make_leaves(n: usize) -> Vec<Hash> {
        (0..n as u64).map(|i| fragment_leaf_hash(i, &[i as u8; 64])).collect()
    }

    #[test]
    fn merkle_root_single_leaf() {
        let leaves = make_leaves(1);
        let root = build_merkle_root(&leaves);
        assert_eq!(root, leaves[0]);
    }

    #[test]
    fn merkle_root_two_leaves() {
        let leaves = make_leaves(2);
        let root = build_merkle_root(&leaves);
        assert_ne!(root, leaves[0]);
        assert_ne!(root, leaves[1]);
    }

    #[test]
    fn merkle_proof_verify_4_leaves() {
        let n = 4usize;
        let fragments: Vec<Vec<u8>> = (0..n).map(|i| vec![i as u8; 64]).collect();
        let leaves: Vec<Hash> = fragments.iter().enumerate().map(|(i, f)| fragment_leaf_hash(i as u64, f)).collect();
        let root = build_merkle_root(&leaves);

        // Build the proof for leaf 2: siblings are [leaf[3], parent(leaf[0],leaf[1])]
        let sibling_leaf = leaves[3];
        let parent_01 = merkle_node_hash(&leaves[0], &leaves[1]);
        let proof = GenomeMerkleProof { leaf_index: 2, siblings: vec![sibling_leaf, parent_01] };

        // Encode root as hex
        let root_bytes: &[u8] = root.as_ref();
        let root_hex: String = root_bytes.iter().map(|b| format!("{b:02x}")).collect();

        assert!(proof.verify(&root_hex, 2, &fragments[2]));
        // Wrong fragment should fail
        assert!(!proof.verify(&root_hex, 2, &fragments[0]));
    }

    #[test]
    fn parse_hash_hex_roundtrip() {
        let h = seed(0xcc);
        let h_bytes: &[u8] = h.as_ref();
        let hex: String = h_bytes.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(parse_hash_hex(&hex), Some(h));
    }

    #[test]
    fn parse_hash_hex_invalid() {
        assert!(parse_hash_hex("").is_none());
        assert!(parse_hash_hex("zz").is_none());
    }

    // ── Fragment cache tests ─────────────────────────────────────────────────

    #[test]
    fn cache_insert_and_get() {
        let cache = GenomeFragmentCache::new(4);
        assert!(cache.is_empty());
        let frag = Arc::new(vec![1u8; 32]);
        cache.insert(7, frag.clone());
        assert_eq!(cache.len(), 1);
        let got = cache.get(7).unwrap();
        assert_eq!(*got, *frag);
    }

    #[test]
    fn cache_evicts_oldest() {
        let cache = GenomeFragmentCache::new(2);
        cache.insert(0, Arc::new(vec![0u8]));
        cache.insert(1, Arc::new(vec![1u8]));
        cache.insert(2, Arc::new(vec![2u8])); // evicts idx 0
        assert!(cache.get(0).is_none());
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());
    }

    // ── Dataset loader tests ─────────────────────────────────────────────────

    #[test]
    fn synthetic_loader_deterministic() {
        let loader = SyntheticLoader::new(256, seed(0x42));
        let a = loader.load_fragment(5).unwrap();
        let b = loader.load_fragment(5).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 256);
    }

    #[test]
    fn synthetic_loader_differs_by_index() {
        let loader = SyntheticLoader::new(256, seed(0x42));
        let a = loader.load_fragment(0).unwrap();
        let b = loader.load_fragment(1).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn cached_loader_caches() {
        let inner = SyntheticLoader::new(64, seed(0x01));
        let cached = CachedLoader::new(inner, 8);
        let a = cached.load_fragment(3).unwrap();
        let b = cached.load_fragment(3).unwrap();
        assert_eq!(a, b);
        assert_eq!(cached.cache.len(), 1);
    }
}
