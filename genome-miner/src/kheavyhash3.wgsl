// KHeavyHash (PyrinHashv2) — Blake3 via array-by-value params (6 args, not 28).
// Passing 28 individual u32s to b3_compress exceeds Metal MSL register limits.

const IV0:u32=0x6A09E667u; const IV1:u32=0xBB67AE85u;
const IV2:u32=0x3C6EF372u; const IV3:u32=0xA54FF53Au;
const IV4:u32=0x510E527Fu; const IV5:u32=0x9B05688Cu;
const IV6:u32=0x1F83D9ABu; const IV7:u32=0x5BE0CD19u;

fn rotr(x:u32,n:u32)->u32{return(x>>n)|(x<<(32u-n));}

// Blake3 compress.  cv and m passed by VALUE (WGSL copies them).
// 6 parameters instead of 28 individual u32s — avoids Metal MSL spill issues.
fn b3_compress(cv:array<u32,8>, m:array<u32,16>, clo:u32, chi:u32, bl:u32, fl:u32)->array<u32,8>{
    var v0=cv[0]; var v1=cv[1]; var v2=cv[2]; var v3=cv[3];
    var v4=cv[4]; var v5=cv[5]; var v6=cv[6]; var v7=cv[7];
    var v8=IV0; var v9=IV1; var v10=IV2; var v11=IV3;
    var v12=clo; var v13=chi; var v14=bl; var v15=fl;
    // Unpack message words to scalar lets — all constant-indexed reads
    let m0=m[0]; let m1=m[1]; let m2=m[2];  let m3=m[3];
    let m4=m[4]; let m5=m[5]; let m6=m[6];  let m7=m[7];
    let m8=m[8]; let m9=m[9]; let m10=m[10]; let m11=m[11];
    let m12=m[12]; let m13=m[13]; let m14=m[14]; let m15=m[15];

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

    var out:array<u32,8>;
    out[0]=v0^v8; out[1]=v1^v9;  out[2]=v2^v10; out[3]=v3^v11;
    out[4]=v4^v12; out[5]=v5^v13; out[6]=v6^v14; out[7]=v7^v15;
    return out;
}

// Blake3 IV constant array (used to construct cv for first compress call)
fn b3_iv()->array<u32,8>{
    var iv:array<u32,8>;
    iv[0]=IV0; iv[1]=IV1; iv[2]=IV2; iv[3]=IV3;
    iv[4]=IV4; iv[5]=IV5; iv[6]=IV6; iv[7]=IV7;
    return iv;
}

// PowHash: blake3(pre_pow_hash[32] || timestamp[8] || zeros[32] || nonce[8])  80 bytes, 2 blocks
// Block1 (64 B, CHUNK_START=1):  ph[0..7] || ts_lo || ts_hi || zeros[6 words]
// Block2 (16 B, CHUNK_END|ROOT=10): zeros[2 words] || nonce_lo || nonce_hi
fn pow_hash(ph:array<u32,8>, ts_lo:u32, ts_hi:u32, nonce_lo:u32, nonce_hi:u32)->array<u32,8>{
    var m1:array<u32,16>;
    m1[0]=ph[0]; m1[1]=ph[1]; m1[2]=ph[2]; m1[3]=ph[3];
    m1[4]=ph[4]; m1[5]=ph[5]; m1[6]=ph[6]; m1[7]=ph[7];
    m1[8]=ts_lo; m1[9]=ts_hi;
    m1[10]=0u; m1[11]=0u; m1[12]=0u; m1[13]=0u; m1[14]=0u; m1[15]=0u;
    let cv=b3_compress(b3_iv(), m1, 0u, 0u, 64u, 1u);

    var m2:array<u32,16>;
    m2[0]=0u; m2[1]=0u; m2[2]=nonce_lo; m2[3]=nonce_hi;
    m2[4]=0u; m2[5]=0u; m2[6]=0u; m2[7]=0u;
    m2[8]=0u; m2[9]=0u; m2[10]=0u; m2[11]=0u; m2[12]=0u; m2[13]=0u; m2[14]=0u; m2[15]=0u;
    return b3_compress(cv, m2, 0u, 0u, 16u, 10u);
}

// KHeavyHash final hash: blake3(product[32]) — single block, flags=CHUNK_START|CHUNK_END|ROOT=11
fn kheavy_final(d:array<u32,8>)->array<u32,8>{
    var m:array<u32,16>;
    m[0]=d[0]; m[1]=d[1]; m[2]=d[2]; m[3]=d[3];
    m[4]=d[4]; m[5]=d[5]; m[6]=d[6]; m[7]=d[7];
    m[8]=0u; m[9]=0u; m[10]=0u; m[11]=0u; m[12]=0u; m[13]=0u; m[14]=0u; m[15]=0u;
    return b3_compress(b3_iv(), m, 0u, 0u, 32u, 11u);
}

// LE 256-bit comparison: return true iff a <= b
fn le256(a:array<u32,8>, b:array<u32,8>)->bool{
    if a[7]<b[7]{return true;} if a[7]>b[7]{return false;}
    if a[6]<b[6]{return true;} if a[6]>b[6]{return false;}
    if a[5]<b[5]{return true;} if a[5]>b[5]{return false;}
    if a[4]<b[4]{return true;} if a[4]>b[4]{return false;}
    if a[3]<b[3]{return true;} if a[3]>b[3]{return false;}
    if a[2]<b[2]{return true;} if a[2]>b[2]{return false;}
    if a[1]<b[1]{return true;} if a[1]>b[1]{return false;}
    if a[0]<b[0]{return true;} if a[0]>b[0]{return false;}
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

    // Step 1: PowHash — blake3(pre_pow_hash || timestamp || zeros[32] || nonce)
    let ph=kparams.pre_pow_hash;
    var h1:array<u32,8>;
    {let t=pow_hash(ph,kparams.timestamp_lo,kparams.timestamp_hi,nonce_lo,nonce_hi);
     h1[0]=t[0];h1[1]=t[1];h1[2]=t[2];h1[3]=t[3];
     h1[4]=t[4];h1[5]=t[5];h1[6]=t[6];h1[7]=t[7];}

    // Step 2a: expand h1 to 64 nibbles (high nibble first per byte)
    var nibbles:array<u32,64>;
    for(var w:u32=0u;w<8u;w++){
        let word=h1[w];
        let b0=word&0xFFu;        let b1=(word>>8u)&0xFFu;
        let b2=(word>>16u)&0xFFu; let b3=(word>>24u)&0xFFu;
        nibbles[w*8u+0u]=(b0>>4u)&0xFu; nibbles[w*8u+1u]=b0&0xFu;
        nibbles[w*8u+2u]=(b1>>4u)&0xFu; nibbles[w*8u+3u]=b1&0xFu;
        nibbles[w*8u+4u]=(b2>>4u)&0xFu; nibbles[w*8u+5u]=b2&0xFu;
        nibbles[w*8u+6u]=(b3>>4u)&0xFu; nibbles[w*8u+7u]=b3&0xFu;
    }

    // Step 2b: matrix-vector multiply + nibble-fold + XOR with h1
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
            pb^=(h1[i]>>(byte_i*8u))&0xFFu;
            out_word|=(pb<<(byte_i*8u));
        }
        prod[i]=out_word;
    }

    // Step 2c: final = blake3(product[32])
    var final_hash:array<u32,8>;
    {let t=kheavy_final(prod);
     final_hash[0]=t[0];final_hash[1]=t[1];final_hash[2]=t[2];final_hash[3]=t[3];
     final_hash[4]=t[4];final_hash[5]=t[5];final_hash[6]=t[6];final_hash[7]=t[7];}

    if le256(final_hash,kparams.pow_target){
        let slot=atomicAdd(&kout.found,1u);
        if slot==0u{
            kout.nonce_lo=nonce_lo;
            kout.nonce_hi=nonce_hi;
            kout.dbg_hash[0]=final_hash[0];
            kout.dbg_hash[1]=final_hash[1];
            kout.dbg_hash[2]=final_hash[2];
            kout.dbg_hash[3]=final_hash[3];
            kout.dbg_hash[4]=final_hash[4];
            kout.dbg_hash[5]=final_hash[5];
            kout.dbg_hash[6]=final_hash[6];
            kout.dbg_hash[7]=final_hash[7];
        }
    }
}
