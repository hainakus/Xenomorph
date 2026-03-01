// KHeavyHash (PyrinHashv2) — Blake3 via var<private> globals.
//
// Previous approaches (28-scalar params, array-by-value params) both produced
// wrong hashes on Metal. Root cause: naga MSL backend silently mis-generates
// code for functions that RETURN array<u32,N>.  Fix: b3_compress writes its
// result into a var<private> global (no array return), takes only 4 scalars.

const IV0:u32=0x6A09E667u; const IV1:u32=0xBB67AE85u;
const IV2:u32=0x3C6EF372u; const IV3:u32=0xA54FF53Au;
const IV4:u32=0x510E527Fu; const IV5:u32=0x9B05688Cu;
const IV6:u32=0x1F83D9ABu; const IV7:u32=0x5BE0CD19u;

// Thread-private Blake3 working state.  b3_compress reads cv from g_state[0..7],
// message from g_msg[0..15], and writes the new cv back to g_state[0..7].
var<private> g_state: array<u32,16>;
var<private> g_msg:   array<u32,16>;

fn rotr(x:u32,n:u32)->u32{return(x>>n)|(x<<(32u-n));}

// Blake3 compression.
//   Reads  : g_state[0..7]  (chaining value), g_msg[0..15] (message)
//   Writes : g_state[0..7]  (new chaining value = v[i] ^ v[i+8])
//   Params : only the 4 scalars that differ per call
fn b3_compress(clo:u32, chi:u32, bl:u32, fl:u32){
    // Load CV and message into scalar locals (all constant-index reads)
    var v0=g_state[0]; var v1=g_state[1]; var v2=g_state[2]; var v3=g_state[3];
    var v4=g_state[4]; var v5=g_state[5]; var v6=g_state[6]; var v7=g_state[7];
    var v8=IV0; var v9=IV1; var v10=IV2; var v11=IV3;
    var v12=clo; var v13=chi; var v14=bl; var v15=fl;
    let m0=g_msg[0];  let m1=g_msg[1];  let m2=g_msg[2];  let m3=g_msg[3];
    let m4=g_msg[4];  let m5=g_msg[5];  let m6=g_msg[6];  let m7=g_msg[7];
    let m8=g_msg[8];  let m9=g_msg[9];  let m10=g_msg[10]; let m11=g_msg[11];
    let m12=g_msg[12]; let m13=g_msg[13]; let m14=g_msg[14]; let m15=g_msg[15];

    // Round 0  sigma=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    v0+=v4+m0;  v12=rotr(v12^v0,16u); v8+=v12;  v4=rotr(v4^v8,12u);  v0+=v4+m1;  v12=rotr(v12^v0,8u);  v8+=v12;  v4=rotr(v4^v8,7u);
    v1+=v5+m2;  v13=rotr(v13^v1,16u); v9+=v13;  v5=rotr(v5^v9,12u);  v1+=v5+m3;  v13=rotr(v13^v1,8u);  v9+=v13;  v5=rotr(v5^v9,7u);
    v2+=v6+m4;  v14=rotr(v14^v2,16u); v10+=v14; v6=rotr(v6^v10,12u); v2+=v6+m5;  v14=rotr(v14^v2,8u);  v10+=v14; v6=rotr(v6^v10,7u);
    v3+=v7+m6;  v15=rotr(v15^v3,16u); v11+=v15; v7=rotr(v7^v11,12u); v3+=v7+m7;  v15=rotr(v15^v3,8u);  v11+=v15; v7=rotr(v7^v11,7u);
    v0+=v5+m8;  v15=rotr(v15^v0,16u); v10+=v15; v5=rotr(v5^v10,12u); v0+=v5+m9;  v15=rotr(v15^v0,8u);  v10+=v15; v5=rotr(v5^v10,7u);
    v1+=v6+m10; v12=rotr(v12^v1,16u); v11+=v12; v6=rotr(v6^v11,12u); v1+=v6+m11; v12=rotr(v12^v1,8u);  v11+=v12; v6=rotr(v6^v11,7u);
    v2+=v7+m12; v13=rotr(v13^v2,16u); v8+=v13;  v7=rotr(v7^v8,12u);  v2+=v7+m13; v13=rotr(v13^v2,8u);  v8+=v13;  v7=rotr(v7^v8,7u);
    v3+=v4+m14; v14=rotr(v14^v3,16u); v9+=v14;  v4=rotr(v4^v9,12u);  v3+=v4+m15; v14=rotr(v14^v3,8u);  v9+=v14;  v4=rotr(v4^v9,7u);
    // Round 1  sigma=[2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
    v0+=v4+m2;  v12=rotr(v12^v0,16u); v8+=v12;  v4=rotr(v4^v8,12u);  v0+=v4+m6;  v12=rotr(v12^v0,8u);  v8+=v12;  v4=rotr(v4^v8,7u);
    v1+=v5+m3;  v13=rotr(v13^v1,16u); v9+=v13;  v5=rotr(v5^v9,12u);  v1+=v5+m10; v13=rotr(v13^v1,8u);  v9+=v13;  v5=rotr(v5^v9,7u);
    v2+=v6+m7;  v14=rotr(v14^v2,16u); v10+=v14; v6=rotr(v6^v10,12u); v2+=v6+m0;  v14=rotr(v14^v2,8u);  v10+=v14; v6=rotr(v6^v10,7u);
    v3+=v7+m4;  v15=rotr(v15^v3,16u); v11+=v15; v7=rotr(v7^v11,12u); v3+=v7+m13; v15=rotr(v15^v3,8u);  v11+=v15; v7=rotr(v7^v11,7u);
    v0+=v5+m1;  v15=rotr(v15^v0,16u); v10+=v15; v5=rotr(v5^v10,12u); v0+=v5+m11; v15=rotr(v15^v0,8u);  v10+=v15; v5=rotr(v5^v10,7u);
    v1+=v6+m12; v12=rotr(v12^v1,16u); v11+=v12; v6=rotr(v6^v11,12u); v1+=v6+m5;  v12=rotr(v12^v1,8u);  v11+=v12; v6=rotr(v6^v11,7u);
    v2+=v7+m9;  v13=rotr(v13^v2,16u); v8+=v13;  v7=rotr(v7^v8,12u);  v2+=v7+m14; v13=rotr(v13^v2,8u);  v8+=v13;  v7=rotr(v7^v8,7u);
    v3+=v4+m15; v14=rotr(v14^v3,16u); v9+=v14;  v4=rotr(v4^v9,12u);  v3+=v4+m8;  v14=rotr(v14^v3,8u);  v9+=v14;  v4=rotr(v4^v9,7u);
    // Round 2  sigma=[3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
    v0+=v4+m3;  v12=rotr(v12^v0,16u); v8+=v12;  v4=rotr(v4^v8,12u);  v0+=v4+m4;  v12=rotr(v12^v0,8u);  v8+=v12;  v4=rotr(v4^v8,7u);
    v1+=v5+m10; v13=rotr(v13^v1,16u); v9+=v13;  v5=rotr(v5^v9,12u);  v1+=v5+m12; v13=rotr(v13^v1,8u);  v9+=v13;  v5=rotr(v5^v9,7u);
    v2+=v6+m13; v14=rotr(v14^v2,16u); v10+=v14; v6=rotr(v6^v10,12u); v2+=v6+m2;  v14=rotr(v14^v2,8u);  v10+=v14; v6=rotr(v6^v10,7u);
    v3+=v7+m7;  v15=rotr(v15^v3,16u); v11+=v15; v7=rotr(v7^v11,12u); v3+=v7+m14; v15=rotr(v15^v3,8u);  v11+=v15; v7=rotr(v7^v11,7u);
    v0+=v5+m6;  v15=rotr(v15^v0,16u); v10+=v15; v5=rotr(v5^v10,12u); v0+=v5+m5;  v15=rotr(v15^v0,8u);  v10+=v15; v5=rotr(v5^v10,7u);
    v1+=v6+m9;  v12=rotr(v12^v1,16u); v11+=v12; v6=rotr(v6^v11,12u); v1+=v6+m0;  v12=rotr(v12^v1,8u);  v11+=v12; v6=rotr(v6^v11,7u);
    v2+=v7+m11; v13=rotr(v13^v2,16u); v8+=v13;  v7=rotr(v7^v8,12u);  v2+=v7+m15; v13=rotr(v13^v2,8u);  v8+=v13;  v7=rotr(v7^v8,7u);
    v3+=v4+m8;  v14=rotr(v14^v3,16u); v9+=v14;  v4=rotr(v4^v9,12u);  v3+=v4+m1;  v14=rotr(v14^v3,8u);  v9+=v14;  v4=rotr(v4^v9,7u);
    // Round 3  sigma=[10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
    v0+=v4+m10; v12=rotr(v12^v0,16u); v8+=v12;  v4=rotr(v4^v8,12u);  v0+=v4+m7;  v12=rotr(v12^v0,8u);  v8+=v12;  v4=rotr(v4^v8,7u);
    v1+=v5+m12; v13=rotr(v13^v1,16u); v9+=v13;  v5=rotr(v5^v9,12u);  v1+=v5+m9;  v13=rotr(v13^v1,8u);  v9+=v13;  v5=rotr(v5^v9,7u);
    v2+=v6+m14; v14=rotr(v14^v2,16u); v10+=v14; v6=rotr(v6^v10,12u); v2+=v6+m3;  v14=rotr(v14^v2,8u);  v10+=v14; v6=rotr(v6^v10,7u);
    v3+=v7+m13; v15=rotr(v15^v3,16u); v11+=v15; v7=rotr(v7^v11,12u); v3+=v7+m15; v15=rotr(v15^v3,8u);  v11+=v15; v7=rotr(v7^v11,7u);
    v0+=v5+m4;  v15=rotr(v15^v0,16u); v10+=v15; v5=rotr(v5^v10,12u); v0+=v5+m0;  v15=rotr(v15^v0,8u);  v10+=v15; v5=rotr(v5^v10,7u);
    v1+=v6+m11; v12=rotr(v12^v1,16u); v11+=v12; v6=rotr(v6^v11,12u); v1+=v6+m2;  v12=rotr(v12^v1,8u);  v11+=v12; v6=rotr(v6^v11,7u);
    v2+=v7+m5;  v13=rotr(v13^v2,16u); v8+=v13;  v7=rotr(v7^v8,12u);  v2+=v7+m8;  v13=rotr(v13^v2,8u);  v8+=v13;  v7=rotr(v7^v8,7u);
    v3+=v4+m1;  v14=rotr(v14^v3,16u); v9+=v14;  v4=rotr(v4^v9,12u);  v3+=v4+m6;  v14=rotr(v14^v3,8u);  v9+=v14;  v4=rotr(v4^v9,7u);
    // Round 4  sigma=[12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
    v0+=v4+m12; v12=rotr(v12^v0,16u); v8+=v12;  v4=rotr(v4^v8,12u);  v0+=v4+m13; v12=rotr(v12^v0,8u);  v8+=v12;  v4=rotr(v4^v8,7u);
    v1+=v5+m9;  v13=rotr(v13^v1,16u); v9+=v13;  v5=rotr(v5^v9,12u);  v1+=v5+m11; v13=rotr(v13^v1,8u);  v9+=v13;  v5=rotr(v5^v9,7u);
    v2+=v6+m15; v14=rotr(v14^v2,16u); v10+=v14; v6=rotr(v6^v10,12u); v2+=v6+m10; v14=rotr(v14^v2,8u);  v10+=v14; v6=rotr(v6^v10,7u);
    v3+=v7+m14; v15=rotr(v15^v3,16u); v11+=v15; v7=rotr(v7^v11,12u); v3+=v7+m8;  v15=rotr(v15^v3,8u);  v11+=v15; v7=rotr(v7^v11,7u);
    v0+=v5+m7;  v15=rotr(v15^v0,16u); v10+=v15; v5=rotr(v5^v10,12u); v0+=v5+m2;  v15=rotr(v15^v0,8u);  v10+=v15; v5=rotr(v5^v10,7u);
    v1+=v6+m5;  v12=rotr(v12^v1,16u); v11+=v12; v6=rotr(v6^v11,12u); v1+=v6+m3;  v12=rotr(v12^v1,8u);  v11+=v12; v6=rotr(v6^v11,7u);
    v2+=v7+m0;  v13=rotr(v13^v2,16u); v8+=v13;  v7=rotr(v7^v8,12u);  v2+=v7+m1;  v13=rotr(v13^v2,8u);  v8+=v13;  v7=rotr(v7^v8,7u);
    v3+=v4+m6;  v14=rotr(v14^v3,16u); v9+=v14;  v4=rotr(v4^v9,12u);  v3+=v4+m4;  v14=rotr(v14^v3,8u);  v9+=v14;  v4=rotr(v4^v9,7u);
    // Round 5  sigma=[9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
    v0+=v4+m9;  v12=rotr(v12^v0,16u); v8+=v12;  v4=rotr(v4^v8,12u);  v0+=v4+m14; v12=rotr(v12^v0,8u);  v8+=v12;  v4=rotr(v4^v8,7u);
    v1+=v5+m11; v13=rotr(v13^v1,16u); v9+=v13;  v5=rotr(v5^v9,12u);  v1+=v5+m5;  v13=rotr(v13^v1,8u);  v9+=v13;  v5=rotr(v5^v9,7u);
    v2+=v6+m8;  v14=rotr(v14^v2,16u); v10+=v14; v6=rotr(v6^v10,12u); v2+=v6+m12; v14=rotr(v14^v2,8u);  v10+=v14; v6=rotr(v6^v10,7u);
    v3+=v7+m15; v15=rotr(v15^v3,16u); v11+=v15; v7=rotr(v7^v11,12u); v3+=v7+m1;  v15=rotr(v15^v3,8u);  v11+=v15; v7=rotr(v7^v11,7u);
    v0+=v5+m13; v15=rotr(v15^v0,16u); v10+=v15; v5=rotr(v5^v10,12u); v0+=v5+m3;  v15=rotr(v15^v0,8u);  v10+=v15; v5=rotr(v5^v10,7u);
    v1+=v6+m0;  v12=rotr(v12^v1,16u); v11+=v12; v6=rotr(v6^v11,12u); v1+=v6+m10; v12=rotr(v12^v1,8u);  v11+=v12; v6=rotr(v6^v11,7u);
    v2+=v7+m2;  v13=rotr(v13^v2,16u); v8+=v13;  v7=rotr(v7^v8,12u);  v2+=v7+m6;  v13=rotr(v13^v2,8u);  v8+=v13;  v7=rotr(v7^v8,7u);
    v3+=v4+m4;  v14=rotr(v14^v3,16u); v9+=v14;  v4=rotr(v4^v9,12u);  v3+=v4+m7;  v14=rotr(v14^v3,8u);  v9+=v14;  v4=rotr(v4^v9,7u);
    // Round 6  sigma=[11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
    v0+=v4+m11; v12=rotr(v12^v0,16u); v8+=v12;  v4=rotr(v4^v8,12u);  v0+=v4+m15; v12=rotr(v12^v0,8u);  v8+=v12;  v4=rotr(v4^v8,7u);
    v1+=v5+m5;  v13=rotr(v13^v1,16u); v9+=v13;  v5=rotr(v5^v9,12u);  v1+=v5+m0;  v13=rotr(v13^v1,8u);  v9+=v13;  v5=rotr(v5^v9,7u);
    v2+=v6+m1;  v14=rotr(v14^v2,16u); v10+=v14; v6=rotr(v6^v10,12u); v2+=v6+m9;  v14=rotr(v14^v2,8u);  v10+=v14; v6=rotr(v6^v10,7u);
    v3+=v7+m8;  v15=rotr(v15^v3,16u); v11+=v15; v7=rotr(v7^v11,12u); v3+=v7+m6;  v15=rotr(v15^v3,8u);  v11+=v15; v7=rotr(v7^v11,7u);
    v0+=v5+m14; v15=rotr(v15^v0,16u); v10+=v15; v5=rotr(v5^v10,12u); v0+=v5+m10; v15=rotr(v15^v0,8u);  v10+=v15; v5=rotr(v5^v10,7u);
    v1+=v6+m2;  v12=rotr(v12^v1,16u); v11+=v12; v6=rotr(v6^v11,12u); v1+=v6+m12; v12=rotr(v12^v1,8u);  v11+=v12; v6=rotr(v6^v11,7u);
    v2+=v7+m3;  v13=rotr(v13^v2,16u); v8+=v13;  v7=rotr(v7^v8,12u);  v2+=v7+m4;  v13=rotr(v13^v2,8u);  v8+=v13;  v7=rotr(v7^v8,7u);
    v3+=v4+m7;  v14=rotr(v14^v3,16u); v9+=v14;  v4=rotr(v4^v9,12u);  v3+=v4+m13; v14=rotr(v14^v3,8u);  v9+=v14;  v4=rotr(v4^v9,7u);

    // Write new chaining value back to g_state[0..7] (constant-index writes)
    g_state[0]=v0^v8;  g_state[1]=v1^v9;  g_state[2]=v2^v10; g_state[3]=v3^v11;
    g_state[4]=v4^v12; g_state[5]=v5^v13; g_state[6]=v6^v14; g_state[7]=v7^v15;
}

// LE 256-bit comparison: return true iff a <= b  (index-by-index, high word first)
fn le256(a0:u32,a1:u32,a2:u32,a3:u32,a4:u32,a5:u32,a6:u32,a7:u32,
         b0:u32,b1:u32,b2:u32,b3:u32,b4:u32,b5:u32,b6:u32,b7:u32)->bool{
    if a7<b7{return true;} if a7>b7{return false;}
    if a6<b6{return true;} if a6>b6{return false;}
    if a5<b5{return true;} if a5>b5{return false;}
    if a4<b4{return true;} if a4>b4{return false;}
    if a3<b3{return true;} if a3>b3{return false;}
    if a2<b2{return true;} if a2>b2{return false;}
    if a1<b1{return true;} if a1>b1{return false;}
    if a0<b0{return true;} if a0>b0{return false;}
    return true;
}

struct KHeavyParams {
    pre_pow_hash:  array<u32,8>,
    timestamp_lo:  u32,
    timestamp_hi:  u32,
    pow_target:    array<u32,8>,
    nonce_base_lo: u32,
    nonce_base_hi: u32,
    pad0: u32,
    pad1: u32,
}
struct KHeavyOutput {
    found:    atomic<u32>,
    nonce_lo: u32,
    nonce_hi: u32,
    pad0:     u32,
    dbg_hash: array<u32,8>,
}

@group(0) @binding(0) var<storage,read>       kparams: KHeavyParams;
@group(0) @binding(1) var<storage,read>        mat:    array<u32>;
@group(0) @binding(2) var<storage,read_write>  kout:   KHeavyOutput;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let delta=gid.x;
    var nonce_lo=kparams.nonce_base_lo+delta;
    var nonce_hi=kparams.nonce_base_hi;
    if nonce_lo<delta{nonce_hi+=1u;}

    // ── Step 1: PowHash = blake3(pre_pow_hash[32] ‖ timestamp[8] ‖ zeros[32] ‖ nonce[8]) ──
    // Block 1 (64 bytes = pre_pow_hash || timestamp || zeros[24]), CHUNK_START flag=1
    g_state[0]=IV0; g_state[1]=IV1; g_state[2]=IV2; g_state[3]=IV3;
    g_state[4]=IV4; g_state[5]=IV5; g_state[6]=IV6; g_state[7]=IV7;
    g_msg[0]=kparams.pre_pow_hash[0]; g_msg[1]=kparams.pre_pow_hash[1];
    g_msg[2]=kparams.pre_pow_hash[2]; g_msg[3]=kparams.pre_pow_hash[3];
    g_msg[4]=kparams.pre_pow_hash[4]; g_msg[5]=kparams.pre_pow_hash[5];
    g_msg[6]=kparams.pre_pow_hash[6]; g_msg[7]=kparams.pre_pow_hash[7];
    g_msg[8]=kparams.timestamp_lo; g_msg[9]=kparams.timestamp_hi;
    g_msg[10]=0u; g_msg[11]=0u; g_msg[12]=0u; g_msg[13]=0u; g_msg[14]=0u; g_msg[15]=0u;
    b3_compress(0u, 0u, 64u, 1u);

    // Block 2 (16 bytes = zeros[8] || nonce[8]), CHUNK_END|ROOT flag=10
    // g_state[0..7] already holds cv from block 1
    g_msg[0]=0u; g_msg[1]=0u; g_msg[2]=nonce_lo; g_msg[3]=nonce_hi;
    g_msg[4]=0u; g_msg[5]=0u; g_msg[6]=0u; g_msg[7]=0u;
    g_msg[8]=0u; g_msg[9]=0u; g_msg[10]=0u; g_msg[11]=0u;
    g_msg[12]=0u; g_msg[13]=0u; g_msg[14]=0u; g_msg[15]=0u;
    b3_compress(0u, 0u, 16u, 10u);

    // pow_hash result is now in g_state[0..7]
    let h0=g_state[0]; let h1=g_state[1]; let h2=g_state[2]; let h3=g_state[3];
    let h4=g_state[4]; let h5=g_state[5]; let h6=g_state[6]; let h7=g_state[7];

    // ── Step 2a: Expand h to 64 nibbles (high nibble first per byte) ──
    var nibbles:array<u32,64>;
    let w0=h0; let w1=h1; let w2=h2; let w3=h3;
    let w4=h4; let w5=h5; let w6=h6; let w7=h7;
    // word 0
    nibbles[0]=(w0>>4u)&0xFu;  nibbles[1]=w0&0xFu;
    nibbles[2]=(w0>>12u)&0xFu; nibbles[3]=(w0>>8u)&0xFu;
    nibbles[4]=(w0>>20u)&0xFu; nibbles[5]=(w0>>16u)&0xFu;
    nibbles[6]=(w0>>28u)&0xFu; nibbles[7]=(w0>>24u)&0xFu;
    // word 1
    nibbles[8]=(w1>>4u)&0xFu;  nibbles[9]=w1&0xFu;
    nibbles[10]=(w1>>12u)&0xFu; nibbles[11]=(w1>>8u)&0xFu;
    nibbles[12]=(w1>>20u)&0xFu; nibbles[13]=(w1>>16u)&0xFu;
    nibbles[14]=(w1>>28u)&0xFu; nibbles[15]=(w1>>24u)&0xFu;
    // word 2
    nibbles[16]=(w2>>4u)&0xFu;  nibbles[17]=w2&0xFu;
    nibbles[18]=(w2>>12u)&0xFu; nibbles[19]=(w2>>8u)&0xFu;
    nibbles[20]=(w2>>20u)&0xFu; nibbles[21]=(w2>>16u)&0xFu;
    nibbles[22]=(w2>>28u)&0xFu; nibbles[23]=(w2>>24u)&0xFu;
    // word 3
    nibbles[24]=(w3>>4u)&0xFu;  nibbles[25]=w3&0xFu;
    nibbles[26]=(w3>>12u)&0xFu; nibbles[27]=(w3>>8u)&0xFu;
    nibbles[28]=(w3>>20u)&0xFu; nibbles[29]=(w3>>16u)&0xFu;
    nibbles[30]=(w3>>28u)&0xFu; nibbles[31]=(w3>>24u)&0xFu;
    // word 4
    nibbles[32]=(w4>>4u)&0xFu;  nibbles[33]=w4&0xFu;
    nibbles[34]=(w4>>12u)&0xFu; nibbles[35]=(w4>>8u)&0xFu;
    nibbles[36]=(w4>>20u)&0xFu; nibbles[37]=(w4>>16u)&0xFu;
    nibbles[38]=(w4>>28u)&0xFu; nibbles[39]=(w4>>24u)&0xFu;
    // word 5
    nibbles[40]=(w5>>4u)&0xFu;  nibbles[41]=w5&0xFu;
    nibbles[42]=(w5>>12u)&0xFu; nibbles[43]=(w5>>8u)&0xFu;
    nibbles[44]=(w5>>20u)&0xFu; nibbles[45]=(w5>>16u)&0xFu;
    nibbles[46]=(w5>>28u)&0xFu; nibbles[47]=(w5>>24u)&0xFu;
    // word 6
    nibbles[48]=(w6>>4u)&0xFu;  nibbles[49]=w6&0xFu;
    nibbles[50]=(w6>>12u)&0xFu; nibbles[51]=(w6>>8u)&0xFu;
    nibbles[52]=(w6>>20u)&0xFu; nibbles[53]=(w6>>16u)&0xFu;
    nibbles[54]=(w6>>28u)&0xFu; nibbles[55]=(w6>>24u)&0xFu;
    // word 7
    nibbles[56]=(w7>>4u)&0xFu;  nibbles[57]=w7&0xFu;
    nibbles[58]=(w7>>12u)&0xFu; nibbles[59]=(w7>>8u)&0xFu;
    nibbles[60]=(w7>>20u)&0xFu; nibbles[61]=(w7>>16u)&0xFu;
    nibbles[62]=(w7>>28u)&0xFu; nibbles[63]=(w7>>24u)&0xFu;

    // ── Step 2b: Matrix-vector multiply + 3-nibble fold + XOR with h ──
    // Copy pow_hash into a var array so we can dynamically index it in the loop
    var pw:array<u32,8>;
    pw[0]=h0; pw[1]=h1; pw[2]=h2; pw[3]=h3;
    pw[4]=h4; pw[5]=h5; pw[6]=h6; pw[7]=h7;
    var prod:array<u32,8>;
    for(var i:u32=0u;i<8u;i++){
        var out_word:u32=0u;
        for(var byte_i:u32=0u;byte_i<4u;byte_i++){
            let out_idx=i*4u+byte_i;
            var sum1:u32=0u; var sum2:u32=0u;
            let row1=out_idx*2u; let row2=out_idx*2u+1u;
            for(var j:u32=0u;j<64u;j++){
                sum1+=mat[row1*64u+j]*nibbles[j];
                sum2+=mat[row2*64u+j]*nibbles[j];
            }
            let n1=((sum1&0xFu)^((sum1>>4u)&0xFu)^((sum1>>8u)&0xFu))&0xFu;
            let n2=((sum2&0xFu)^((sum2>>4u)&0xFu)^((sum2>>8u)&0xFu))&0xFu;
            var pb=(n1<<4u)|n2;
            pb^=(pw[i]>>(byte_i*8u))&0xFFu;
            out_word|=(pb<<(byte_i*8u));
        }
        prod[i]=out_word;
    }

    // ── Step 2c: KHeavyHash final = blake3(product[32]) single block, flags=11 ──
    g_state[0]=IV0; g_state[1]=IV1; g_state[2]=IV2; g_state[3]=IV3;
    g_state[4]=IV4; g_state[5]=IV5; g_state[6]=IV6; g_state[7]=IV7;
    g_msg[0]=prod[0]; g_msg[1]=prod[1]; g_msg[2]=prod[2]; g_msg[3]=prod[3];
    g_msg[4]=prod[4]; g_msg[5]=prod[5]; g_msg[6]=prod[6]; g_msg[7]=prod[7];
    g_msg[8]=0u; g_msg[9]=0u; g_msg[10]=0u; g_msg[11]=0u;
    g_msg[12]=0u; g_msg[13]=0u; g_msg[14]=0u; g_msg[15]=0u;
    b3_compress(0u, 0u, 32u, 11u);

    let f0=g_state[0]; let f1=g_state[1]; let f2=g_state[2]; let f3=g_state[3];
    let f4=g_state[4]; let f5=g_state[5]; let f6=g_state[6]; let f7=g_state[7];

    if le256(f0,f1,f2,f3,f4,f5,f6,f7,
             kparams.pow_target[0],kparams.pow_target[1],kparams.pow_target[2],kparams.pow_target[3],
             kparams.pow_target[4],kparams.pow_target[5],kparams.pow_target[6],kparams.pow_target[7]){
        let slot=atomicAdd(&kout.found,1u);
        if slot==0u{
            kout.nonce_lo=nonce_lo;
            kout.nonce_hi=nonce_hi;
            kout.dbg_hash[0]=f0; kout.dbg_hash[1]=f1; kout.dbg_hash[2]=f2; kout.dbg_hash[3]=f3;
            kout.dbg_hash[4]=f4; kout.dbg_hash[5]=f5; kout.dbg_hash[6]=f6; kout.dbg_hash[7]=f7;
        }
    }
}
