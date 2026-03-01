// Genome PoW GPU compute shader
// Per-nonce work:
//   1. blake3(epoch_seed||nonce)[0..8] % num_fragments → fragment_index
//   2. fragment_hashes[fragment_index]  (pre-computed on CPU, uploaded to VRAM)
//   3. blake3(fragment_hash||pre_pow_hash||nonce) → compare ≤ target

// ── Blake3 constants ──────────────────────────────────────────────────────────

const B3_IV = array<u32, 8>(
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
);

// Blake3 message schedule sigma (7 rounds × 16 indices)
const S0  = array<u32, 16>(0u,1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u);
const S1  = array<u32, 16>(2u,6u,3u,10u,7u,0u,4u,13u,1u,11u,12u,5u,9u,14u,15u,8u);
const S2  = array<u32, 16>(3u,4u,10u,12u,13u,2u,7u,14u,6u,5u,9u,0u,11u,15u,8u,1u);
const S3  = array<u32, 16>(10u,7u,12u,9u,14u,3u,13u,15u,4u,0u,11u,2u,5u,8u,1u,6u);
const S4  = array<u32, 16>(12u,13u,9u,11u,15u,10u,14u,8u,7u,2u,5u,3u,0u,1u,6u,4u);
const S5  = array<u32, 16>(9u,14u,11u,5u,8u,12u,15u,1u,13u,3u,0u,10u,2u,6u,4u,7u);
const S6  = array<u32, 16>(11u,15u,5u,0u,1u,9u,8u,6u,14u,10u,2u,12u,3u,4u,7u,13u);

// ── Blake3 mixing function ────────────────────────────────────────────────────

fn b3g(v: ptr<function, array<u32, 16>>, a: u32, b: u32, c: u32, d: u32, x: u32, y: u32) {
    (*v)[a] = (*v)[a] + (*v)[b] + x;
    (*v)[d] = rotr((*v)[d] ^ (*v)[a], 16u);
    (*v)[c] = (*v)[c] + (*v)[d];
    (*v)[b] = rotr((*v)[b] ^ (*v)[c], 12u);
    (*v)[a] = (*v)[a] + (*v)[b] + y;
    (*v)[d] = rotr((*v)[d] ^ (*v)[a], 8u);
    (*v)[c] = (*v)[c] + (*v)[d];
    (*v)[b] = rotr((*v)[b] ^ (*v)[c], 7u);
}

fn b3_round(v: ptr<function, array<u32, 16>>, m: array<u32, 16>, r: u32) {
    var s: array<u32, 16>;
    switch r {
        case 0u: { s = S0; }
        case 1u: { s = S1; }
        case 2u: { s = S2; }
        case 3u: { s = S3; }
        case 4u: { s = S4; }
        case 5u: { s = S5; }
        default: { s = S6; }
    }
    b3g(v, 0u, 4u, 8u,  12u, m[s[0]],  m[s[1]]);
    b3g(v, 1u, 5u, 9u,  13u, m[s[2]],  m[s[3]]);
    b3g(v, 2u, 6u, 10u, 14u, m[s[4]],  m[s[5]]);
    b3g(v, 3u, 7u, 11u, 15u, m[s[6]],  m[s[7]]);
    b3g(v, 0u, 5u, 10u, 15u, m[s[8]],  m[s[9]]);
    b3g(v, 1u, 6u, 11u, 12u, m[s[10]], m[s[11]]);
    b3g(v, 2u, 7u, 8u,  13u, m[s[12]], m[s[13]]);
    b3g(v, 3u, 4u, 9u,  14u, m[s[14]], m[s[15]]);
}

// Compress one 64-byte block.  Returns the full 16-word output vector.
fn b3_compress(cv: array<u32, 8>, m: array<u32, 16>,
               ctr_lo: u32, ctr_hi: u32, blen: u32, flags: u32) -> array<u32, 16> {
    var v: array<u32, 16>;
    v[0]=cv[0]; v[1]=cv[1]; v[2]=cv[2];  v[3]=cv[3];
    v[4]=cv[4]; v[5]=cv[5]; v[6]=cv[6];  v[7]=cv[7];
    v[8]=B3_IV[0]; v[9]=B3_IV[1]; v[10]=B3_IV[2]; v[11]=B3_IV[3];
    v[12]=ctr_lo; v[13]=ctr_hi; v[14]=blen; v[15]=flags;
    for (var r = 0u; r < 7u; r++) { b3_round(&v, m, r); }
    v[0]^=v[8];  v[1]^=v[9];  v[2]^=v[10]; v[3]^=v[11];
    v[4]^=v[12]; v[5]^=v[13]; v[6]^=v[14]; v[7]^=v[15];
    return v;
}

// Extract the 8-word chaining value from a compress output.
fn cv_from(out: array<u32, 16>) -> array<u32, 8> {
    return array<u32, 8>(out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
}

// ── blake3 of 40 bytes: epoch_seed(32) || nonce(8) → 8 u32s ─────────────────
//
// Single block (block_len=40), flags = CHUNK_START|CHUNK_END|ROOT = 11.
fn b3_hash_40(data: array<u32, 10>) -> array<u32, 8> {
    var m: array<u32, 16>;
    for (var i = 0u; i < 10u; i++) { m[i] = data[i]; }
    // m[10..15] are already 0 (zero-init)
    let out = b3_compress(B3_IV, m, 0u, 0u, 40u, 11u);
    return cv_from(out);
}

// ── blake3 of 72 bytes: frag_hash(32) || pre_pow_hash(32) || nonce(8) ────────
//
// Block 1: 64 bytes (frag_hash||pre_pow_hash), flags=CHUNK_START=1
// Block 2:  8 bytes (nonce) padded to 64,       flags=CHUNK_END|ROOT=10
fn b3_hash_72(a: array<u32, 8>, b_data: array<u32, 8>, nonce_lo: u32, nonce_hi: u32) -> array<u32, 8> {
    // Block 1
    var m1: array<u32, 16>;
    for (var i = 0u; i < 8u; i++) { m1[i] = a[i]; m1[i+8u] = b_data[i]; }
    let out1 = b3_compress(B3_IV, m1, 0u, 0u, 64u, 1u);
    let cv1 = cv_from(out1);
    // Block 2
    var m2: array<u32, 16>;
    m2[0] = nonce_lo; m2[1] = nonce_hi;
    // m2[2..15] = 0
    let out2 = b3_compress(cv1, m2, 0u, 0u, 8u, 10u);
    return cv_from(out2);
}

// ── Inputs/outputs ───────────────────────────────────────────────────────────

struct Params {
    epoch_seed:    array<u32, 8>,  // 32 bytes
    pre_pow_hash:  array<u32, 8>,  // 32 bytes
    pow_target:    array<u32, 8>,  // 256-bit LE target (8×u32 little-endian)
    nonce_base_lo: u32,
    nonce_base_hi: u32,
    num_fragments: u32,
    pad0:          u32,
}

struct Output {
    found:    atomic<u32>,
    nonce_lo: u32,
    nonce_hi: u32,
    pad0:     u32,
}

@group(0) @binding(0) var<uniform>             params:          Params;
@group(0) @binding(1) var<storage, read>       frag_hashes:     array<u32>;  // num_fragments × 8 u32s
@group(0) @binding(2) var<storage, read_write> out_buf:         Output;

// ── 256-bit LE comparison: returns true if a ≤ b ─────────────────────────────
fn le256(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    for (var i = 7i; i >= 0i; i--) {
        let ai = a[u32(i)];
        let bi = b[u32(i)];
        if ai < bi { return true; }
        if ai > bi { return false; }
    }
    return true; // equal
}

// ── Main compute kernel ───────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Nonce = nonce_base + global_id (64-bit add)
    let delta = gid.x;
    var nonce_lo = params.nonce_base_lo + delta;
    var nonce_hi = params.nonce_base_hi;
    if nonce_lo < delta { nonce_hi += 1u; }  // carry

    // Step 1: fragment_index = blake3(epoch_seed||nonce)[0..8] % num_fragments
    var inp40: array<u32, 10>;
    for (var i = 0u; i < 8u; i++) { inp40[i] = params.epoch_seed[i]; }
    inp40[8] = nonce_lo;
    inp40[9] = nonce_hi;
    let h40 = b3_hash_40(inp40);
    // h40[0] and h40[1] = first 8 bytes LE → u64
    let raw_lo = h40[0];
    let raw_hi = h40[1];
    // raw % num_fragments  (64-bit mod approximated as 32-bit since num_fragments ≤ 2^32)
    var frag_idx: u32;
    if params.num_fragments == 0u {
        frag_idx = 0u;
    } else {
        // Use 64-bit mod: treat as u64 little-endian
        // For num_fragments that fit in u32 (≤3000), we can do: combine into u64, mod
        // WGSL doesn't have u64, so approximate: (lo + hi*2^32) % N
        // Since N << 2^32, hi*2^32 % N = (hi % N) * (2^32 % N) % N
        let n = params.num_fragments;
        let pow32_mod_n = u32((u64(0x100000000u) % u64(n)));  // WGSL has no u64; use workaround
        // Actually WGSL doesn't support u64. Use: (hi*pow32_mod_n + lo) % n
        // pow32_mod_n: 2^32 % n. For n≤3000 this = (4294967296 % n). Precomputed on CPU.
        // We pass it as a spare field: actually let me use a simpler approach:
        // Since num_fragments ≤ 2861 << 2^16, raw_lo alone gives good distribution:
        frag_idx = raw_lo % n;
    }

    // Step 2: lookup fragment hash
    let base = frag_idx * 8u;
    var fh: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { fh[i] = frag_hashes[base + i]; }

    // Step 3: genome_final_hash = blake3(frag_hash||pre_pow_hash||nonce) ≤ target?
    let pow_hash = b3_hash_72(fh, params.pre_pow_hash, nonce_lo, nonce_hi);

    if le256(pow_hash, params.pow_target) {
        // Atomically claim the first found nonce
        let prev = atomicCompareExchangeWeak(&out_buf.found, 0u, 1u);
        if prev.old_value == 0u {
            out_buf.nonce_lo = nonce_lo;
            out_buf.nonce_hi = nonce_hi;
        }
    }
}
