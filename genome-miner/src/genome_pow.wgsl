// Genome PoW GPU compute shader  — memory-hard algorithm
//
// Per-nonce work (8 × 32-byte random reads from the full 739 MB packed genome):
//   state  = blake3_40(epoch_seed || nonce)
//   for 8 rounds:
//     pos    = state[0..2] as u64 % num_mix_chunks
//     chunk  = packed_genome[pos*8 .. pos*8+8]   (8 u32 = 32 bytes)
//     state  = blake3_64(state, chunk)
//   pow    = blake3_72(state || pre_pow_hash || nonce)  → compare ≤ target
//
// Miners must hold all 739 MB in GPU VRAM; random positions make pre-computation impossible.

// ── Blake3 IV ─────────────────────────────────────────────────────────────────

const IV0 : u32 = 0x6A09E667u;
const IV1 : u32 = 0xBB67AE85u;
const IV2 : u32 = 0x3C6EF372u;
const IV3 : u32 = 0xA54FF53Au;
const IV4 : u32 = 0x510E527Fu;
const IV5 : u32 = 0x9B05688Cu;
const IV6 : u32 = 0x1F83D9ABu;
const IV7 : u32 = 0x5BE0CD19u;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }

// ── Blake3 G mixing function ──────────────────────────────────────────────────

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

// ── Compress: 7 rounds fully inlined (avoids runtime double-array indexing) ───

fn b3_compress(cv0: u32, cv1: u32, cv2: u32, cv3: u32,
               cv4: u32, cv5: u32, cv6: u32, cv7: u32,
               m: ptr<function, array<u32, 16>>,
               ctr_lo: u32, ctr_hi: u32, blen: u32, flags: u32)
               -> array<u32, 8> {
    var v: array<u32, 16>;
    v[0]=cv0; v[1]=cv1; v[2]=cv2; v[3]=cv3;
    v[4]=cv4; v[5]=cv5; v[6]=cv6; v[7]=cv7;
    v[8]=IV0; v[9]=IV1; v[10]=IV2; v[11]=IV3;
    v[12]=ctr_lo; v[13]=ctr_hi; v[14]=blen; v[15]=flags;

    // Round 0  sigma=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    b3g(&v,0u,4u,8u, 12u,(*m)[0], (*m)[1]);
    b3g(&v,1u,5u,9u, 13u,(*m)[2], (*m)[3]);
    b3g(&v,2u,6u,10u,14u,(*m)[4], (*m)[5]);
    b3g(&v,3u,7u,11u,15u,(*m)[6], (*m)[7]);
    b3g(&v,0u,5u,10u,15u,(*m)[8], (*m)[9]);
    b3g(&v,1u,6u,11u,12u,(*m)[10],(*m)[11]);
    b3g(&v,2u,7u,8u, 13u,(*m)[12],(*m)[13]);
    b3g(&v,3u,4u,9u, 14u,(*m)[14],(*m)[15]);

    // Round 1  sigma=[2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
    b3g(&v,0u,4u,8u, 12u,(*m)[2], (*m)[6]);
    b3g(&v,1u,5u,9u, 13u,(*m)[3], (*m)[10]);
    b3g(&v,2u,6u,10u,14u,(*m)[7], (*m)[0]);
    b3g(&v,3u,7u,11u,15u,(*m)[4], (*m)[13]);
    b3g(&v,0u,5u,10u,15u,(*m)[1], (*m)[11]);
    b3g(&v,1u,6u,11u,12u,(*m)[12],(*m)[5]);
    b3g(&v,2u,7u,8u, 13u,(*m)[9], (*m)[14]);
    b3g(&v,3u,4u,9u, 14u,(*m)[15],(*m)[8]);

    // Round 2  sigma=[3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
    b3g(&v,0u,4u,8u, 12u,(*m)[3], (*m)[4]);
    b3g(&v,1u,5u,9u, 13u,(*m)[10],(*m)[12]);
    b3g(&v,2u,6u,10u,14u,(*m)[13],(*m)[2]);
    b3g(&v,3u,7u,11u,15u,(*m)[7], (*m)[14]);
    b3g(&v,0u,5u,10u,15u,(*m)[6], (*m)[5]);
    b3g(&v,1u,6u,11u,12u,(*m)[9], (*m)[0]);
    b3g(&v,2u,7u,8u, 13u,(*m)[11],(*m)[15]);
    b3g(&v,3u,4u,9u, 14u,(*m)[8], (*m)[1]);

    // Round 3  sigma=[10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
    b3g(&v,0u,4u,8u, 12u,(*m)[10],(*m)[7]);
    b3g(&v,1u,5u,9u, 13u,(*m)[12],(*m)[9]);
    b3g(&v,2u,6u,10u,14u,(*m)[14],(*m)[3]);
    b3g(&v,3u,7u,11u,15u,(*m)[13],(*m)[15]);
    b3g(&v,0u,5u,10u,15u,(*m)[4], (*m)[0]);
    b3g(&v,1u,6u,11u,12u,(*m)[11],(*m)[2]);
    b3g(&v,2u,7u,8u, 13u,(*m)[5], (*m)[8]);
    b3g(&v,3u,4u,9u, 14u,(*m)[1], (*m)[6]);

    // Round 4  sigma=[12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
    b3g(&v,0u,4u,8u, 12u,(*m)[12],(*m)[13]);
    b3g(&v,1u,5u,9u, 13u,(*m)[9], (*m)[11]);
    b3g(&v,2u,6u,10u,14u,(*m)[15],(*m)[10]);
    b3g(&v,3u,7u,11u,15u,(*m)[14],(*m)[8]);
    b3g(&v,0u,5u,10u,15u,(*m)[7], (*m)[2]);
    b3g(&v,1u,6u,11u,12u,(*m)[5], (*m)[3]);
    b3g(&v,2u,7u,8u, 13u,(*m)[0], (*m)[1]);
    b3g(&v,3u,4u,9u, 14u,(*m)[6], (*m)[4]);

    // Round 5  sigma=[9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
    b3g(&v,0u,4u,8u, 12u,(*m)[9], (*m)[14]);
    b3g(&v,1u,5u,9u, 13u,(*m)[11],(*m)[5]);
    b3g(&v,2u,6u,10u,14u,(*m)[8], (*m)[12]);
    b3g(&v,3u,7u,11u,15u,(*m)[15],(*m)[1]);
    b3g(&v,0u,5u,10u,15u,(*m)[13],(*m)[3]);
    b3g(&v,1u,6u,11u,12u,(*m)[0], (*m)[10]);
    b3g(&v,2u,7u,8u, 13u,(*m)[2], (*m)[6]);
    b3g(&v,3u,4u,9u, 14u,(*m)[4], (*m)[7]);

    // Round 6  sigma=[11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
    b3g(&v,0u,4u,8u, 12u,(*m)[11],(*m)[15]);
    b3g(&v,1u,5u,9u, 13u,(*m)[5], (*m)[0]);
    b3g(&v,2u,6u,10u,14u,(*m)[1], (*m)[9]);
    b3g(&v,3u,7u,11u,15u,(*m)[8], (*m)[6]);
    b3g(&v,0u,5u,10u,15u,(*m)[14],(*m)[10]);
    b3g(&v,1u,6u,11u,12u,(*m)[2], (*m)[12]);
    b3g(&v,2u,7u,8u, 13u,(*m)[3], (*m)[4]);
    b3g(&v,3u,4u,9u, 14u,(*m)[7], (*m)[13]);

    v[0]^=v[8];  v[1]^=v[9];  v[2]^=v[10]; v[3]^=v[11];
    v[4]^=v[12]; v[5]^=v[13]; v[6]^=v[14]; v[7]^=v[15];
    return array<u32,8>(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
}

// ── blake3 of 40 bytes: epoch_seed(32) || nonce(8) → 8 u32s ─────────────────
fn b3_hash_40(d0: u32, d1: u32, d2: u32, d3: u32, d4: u32, d5: u32, d6: u32, d7: u32,
              d8: u32, d9: u32) -> array<u32, 8> {
    var m: array<u32, 16>;
    m[0]=d0; m[1]=d1; m[2]=d2; m[3]=d3; m[4]=d4;
    m[5]=d5; m[6]=d6; m[7]=d7; m[8]=d8; m[9]=d9;
    return b3_compress(IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7, &m, 0u,0u, 40u, 11u);
}

// ── blake3 of 64 bytes: state(32) || chunk(32) — single block ──────────────────
// flags = CHUNK_START(1) | CHUNK_END(2) | ROOT(8) = 11
fn b3_hash_64(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var m: array<u32, 16>;
    m[0]=a[0]; m[1]=a[1]; m[2]=a[2];  m[3]=a[3];
    m[4]=a[4]; m[5]=a[5]; m[6]=a[6];  m[7]=a[7];
    m[8]=b[0]; m[9]=b[1]; m[10]=b[2]; m[11]=b[3];
    m[12]=b[4];m[13]=b[5];m[14]=b[6]; m[15]=b[7];
    return b3_compress(IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7, &m, 0u,0u, 64u, 11u);
}

// ── blake3 of 72 bytes: state(32)||pre_pow_hash(32)||nonce(8) ────────────────
fn b3_hash_72(fh: array<u32, 8>, ph: array<u32, 8>, nonce_lo: u32, nonce_hi: u32) -> array<u32, 8> {
    // Block 1: 64 bytes (frag_hash||pre_pow_hash), flags=CHUNK_START=1
    var m1: array<u32, 16>;
    m1[0]=fh[0]; m1[1]=fh[1]; m1[2]=fh[2]; m1[3]=fh[3];
    m1[4]=fh[4]; m1[5]=fh[5]; m1[6]=fh[6]; m1[7]=fh[7];
    m1[8]=ph[0]; m1[9]=ph[1]; m1[10]=ph[2]; m1[11]=ph[3];
    m1[12]=ph[4]; m1[13]=ph[5]; m1[14]=ph[6]; m1[15]=ph[7];
    let cv1 = b3_compress(IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7, &m1, 0u,0u, 64u, 1u);

    // Block 2: 8 bytes (nonce), flags=CHUNK_END|ROOT=10
    var m2: array<u32, 16>;
    m2[0]=nonce_lo; m2[1]=nonce_hi;
    return b3_compress(cv1[0],cv1[1],cv1[2],cv1[3],cv1[4],cv1[5],cv1[6],cv1[7],
                       &m2, 0u,0u, 8u, 10u);
}

// ── Inputs/outputs ───────────────────────────────────────────────────────────

struct Params {
    epoch_seed:     array<u32, 8>,  // 32 bytes
    pre_pow_hash:   array<u32, 8>,  // 32 bytes
    pow_target:     array<u32, 8>,  // 256-bit LE target (8×u32 little-endian)
    nonce_base_lo:  u32,
    nonce_base_hi:  u32,
    num_mix_chunks: u32,            // packed_genome_bytes / 32
    pad0:           u32,
}

struct Output {
    found:    atomic<u32>,
    nonce_lo: u32,
    nonce_hi: u32,
    pad0:     u32,
}

@group(0) @binding(0) var<storage, read>        params:        Params;
@group(0) @binding(1) var<storage, read>        packed_genome: array<u32>;  // full 739 MB packed dataset
@group(0) @binding(2) var<storage, read_write>  out_buf:       Output;

// ── 256-bit LE comparison: returns true if a ≤ b ─────────────────────────────
fn le256(a: array<u32, 8>, b: array<u32, 8>) -> bool {
    if a[7] < b[7] { return true; } if a[7] > b[7] { return false; }
    if a[6] < b[6] { return true; } if a[6] > b[6] { return false; }
    if a[5] < b[5] { return true; } if a[5] > b[5] { return false; }
    if a[4] < b[4] { return true; } if a[4] > b[4] { return false; }
    if a[3] < b[3] { return true; } if a[3] > b[3] { return false; }
    if a[2] < b[2] { return true; } if a[2] > b[2] { return false; }
    if a[1] < b[1] { return true; } if a[1] > b[1] { return false; }
    if a[0] < b[0] { return true; } if a[0] > b[0] { return false; }
    return true;
}

// ── Main compute kernel ───────────────────────────────────────────────────────

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Nonce = nonce_base + global_id (64-bit add)
    let delta = gid.x;
    var nonce_lo = params.nonce_base_lo + delta;
    var nonce_hi = params.nonce_base_hi;
    if nonce_lo < delta { nonce_hi += 1u; }  // carry

    // Step 1: initial state = blake3(epoch_seed || nonce)
    var state: array<u32, 8> = b3_hash_40(
        params.epoch_seed[0], params.epoch_seed[1],
        params.epoch_seed[2], params.epoch_seed[3],
        params.epoch_seed[4], params.epoch_seed[5],
        params.epoch_seed[6], params.epoch_seed[7],
        nonce_lo, nonce_hi);

    // Step 2: 8 mixing rounds — random 32-byte reads from full packed genome.
    // pos_idx uses 64-bit mod in 32-bit arithmetic (same pattern as old frag_idx).
    // packed_genome is array<u32>, so 8 u32s = 32 bytes per chunk.
    let n = params.num_mix_chunks;
    if n > 0u {
        for (var i = 0u; i < 8u; i++) {
            let p   = (0xFFFFFFFFu % n + 1u) % n;
            let pos = (state[1] % n * p + state[0] % n) % n;
            let base = pos * 8u;
            let chunk = array<u32, 8>(
                packed_genome[base],      packed_genome[base + 1u],
                packed_genome[base + 2u], packed_genome[base + 3u],
                packed_genome[base + 4u], packed_genome[base + 5u],
                packed_genome[base + 6u], packed_genome[base + 7u]);
            state = b3_hash_64(state, chunk);
        }
    }

    // Step 3: pow = blake3(state || pre_pow_hash || nonce) ≤ target?
    let pow_hash = b3_hash_72(state, params.pre_pow_hash, nonce_lo, nonce_hi);

    if le256(pow_hash, params.pow_target) {
        // atomicAdd returns the OLD value; the invocation that gets 0 is first winner
        let slot = atomicAdd(&out_buf.found, 1u);
        if slot == 0u {
            out_buf.nonce_lo = nonce_lo;
            out_buf.nonce_hi = nonce_hi;
        }
    }
}
