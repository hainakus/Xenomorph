// KHeavyHash (PyrinHashv2) GPU compute shader
// Per-nonce:
//   1. PowHash  = blake3(pre_pow_hash[32] || timestamp[8] || zeros[32] || nonce[8]) -> 32 B
//   2. HeavyHash:
//      a. expand PowHash to 64 nibbles
//      b. matrix[64x64] * nibbles -> 32 product bytes (nibble fold + XOR)
//      c. result = blake3(product[32])
//   3. result ≤ target  -> winner

// ── Blake3 IV / helpers ────────────────────────────────────────────────────────
const IV0:u32=0x6A09E667u; const IV1:u32=0xBB67AE85u;
const IV2:u32=0x3C6EF372u; const IV3:u32=0xA54FF53Au;
const IV4:u32=0x510E527Fu; const IV5:u32=0x9B05688Cu;
const IV6:u32=0x1F83D9ABu; const IV7:u32=0x5BE0CD19u;

fn rotr(x:u32,n:u32)->u32{return(x>>n)|(x<<(32u-n));}

fn b3g(v:ptr<function,array<u32,16>>,a:u32,b:u32,c:u32,d:u32,x:u32,y:u32){
    (*v)[a]=(*v)[a]+(*v)[b]+x; (*v)[d]=rotr((*v)[d]^(*v)[a],16u);
    (*v)[c]=(*v)[c]+(*v)[d];   (*v)[b]=rotr((*v)[b]^(*v)[c],12u);
    (*v)[a]=(*v)[a]+(*v)[b]+y; (*v)[d]=rotr((*v)[d]^(*v)[a],8u);
    (*v)[c]=(*v)[c]+(*v)[d];   (*v)[b]=rotr((*v)[b]^(*v)[c],7u);
}

fn b3_compress(c0:u32,c1:u32,c2:u32,c3:u32,c4:u32,c5:u32,c6:u32,c7:u32,
               m:ptr<function,array<u32,16>>,clo:u32,chi:u32,bl:u32,fl:u32)->array<u32,8>{
    var v:array<u32,16>;
    v[0]=c0;v[1]=c1;v[2]=c2;v[3]=c3;v[4]=c4;v[5]=c5;v[6]=c6;v[7]=c7;
    v[8]=IV0;v[9]=IV1;v[10]=IV2;v[11]=IV3;v[12]=clo;v[13]=chi;v[14]=bl;v[15]=fl;
    b3g(&v,0u,4u,8u, 12u,(*m)[0], (*m)[1]); b3g(&v,1u,5u,9u, 13u,(*m)[2], (*m)[3]);
    b3g(&v,2u,6u,10u,14u,(*m)[4], (*m)[5]); b3g(&v,3u,7u,11u,15u,(*m)[6], (*m)[7]);
    b3g(&v,0u,5u,10u,15u,(*m)[8], (*m)[9]); b3g(&v,1u,6u,11u,12u,(*m)[10],(*m)[11]);
    b3g(&v,2u,7u,8u, 13u,(*m)[12],(*m)[13]);b3g(&v,3u,4u,9u, 14u,(*m)[14],(*m)[15]);
    b3g(&v,0u,4u,8u, 12u,(*m)[2], (*m)[6]); b3g(&v,1u,5u,9u, 13u,(*m)[3], (*m)[10]);
    b3g(&v,2u,6u,10u,14u,(*m)[7], (*m)[0]); b3g(&v,3u,7u,11u,15u,(*m)[4], (*m)[13]);
    b3g(&v,0u,5u,10u,15u,(*m)[1], (*m)[11]);b3g(&v,1u,6u,11u,12u,(*m)[12],(*m)[5]);
    b3g(&v,2u,7u,8u, 13u,(*m)[9], (*m)[14]);b3g(&v,3u,4u,9u, 14u,(*m)[15],(*m)[8]);
    b3g(&v,0u,4u,8u, 12u,(*m)[3], (*m)[4]); b3g(&v,1u,5u,9u, 13u,(*m)[10],(*m)[12]);
    b3g(&v,2u,6u,10u,14u,(*m)[13],(*m)[2]); b3g(&v,3u,7u,11u,15u,(*m)[7], (*m)[14]);
    b3g(&v,0u,5u,10u,15u,(*m)[6], (*m)[5]); b3g(&v,1u,6u,11u,12u,(*m)[9], (*m)[0]);
    b3g(&v,2u,7u,8u, 13u,(*m)[11],(*m)[15]);b3g(&v,3u,4u,9u, 14u,(*m)[8], (*m)[1]);
    b3g(&v,0u,4u,8u, 12u,(*m)[10],(*m)[7]); b3g(&v,1u,5u,9u, 13u,(*m)[12],(*m)[9]);
    b3g(&v,2u,6u,10u,14u,(*m)[14],(*m)[3]); b3g(&v,3u,7u,11u,15u,(*m)[13],(*m)[15]);
    b3g(&v,0u,5u,10u,15u,(*m)[4], (*m)[0]); b3g(&v,1u,6u,11u,12u,(*m)[11],(*m)[2]);
    b3g(&v,2u,7u,8u, 13u,(*m)[5], (*m)[8]); b3g(&v,3u,4u,9u, 14u,(*m)[1], (*m)[6]);
    b3g(&v,0u,4u,8u, 12u,(*m)[12],(*m)[13]);b3g(&v,1u,5u,9u, 13u,(*m)[9], (*m)[11]);
    b3g(&v,2u,6u,10u,14u,(*m)[15],(*m)[10]);b3g(&v,3u,7u,11u,15u,(*m)[14],(*m)[8]);
    b3g(&v,0u,5u,10u,15u,(*m)[7], (*m)[2]); b3g(&v,1u,6u,11u,12u,(*m)[5], (*m)[3]);
    b3g(&v,2u,7u,8u, 13u,(*m)[0], (*m)[1]); b3g(&v,3u,4u,9u, 14u,(*m)[6], (*m)[4]);
    b3g(&v,0u,4u,8u, 12u,(*m)[9], (*m)[14]);b3g(&v,1u,5u,9u, 13u,(*m)[11],(*m)[5]);
    b3g(&v,2u,6u,10u,14u,(*m)[8], (*m)[12]);b3g(&v,3u,7u,11u,15u,(*m)[15],(*m)[1]);
    b3g(&v,0u,5u,10u,15u,(*m)[13],(*m)[3]); b3g(&v,1u,6u,11u,12u,(*m)[0], (*m)[10]);
    b3g(&v,2u,7u,8u, 13u,(*m)[2], (*m)[6]); b3g(&v,3u,4u,9u, 14u,(*m)[4], (*m)[7]);
    b3g(&v,0u,4u,8u, 12u,(*m)[11],(*m)[15]);b3g(&v,1u,5u,9u, 13u,(*m)[5], (*m)[0]);
    b3g(&v,2u,6u,10u,14u,(*m)[1], (*m)[9]); b3g(&v,3u,7u,11u,15u,(*m)[8], (*m)[6]);
    b3g(&v,0u,5u,10u,15u,(*m)[14],(*m)[10]);b3g(&v,1u,6u,11u,12u,(*m)[2], (*m)[12]);
    b3g(&v,2u,7u,8u, 13u,(*m)[3], (*m)[4]); b3g(&v,3u,4u,9u, 14u,(*m)[7], (*m)[13]);
    v[0]^=v[8];v[1]^=v[9];v[2]^=v[10];v[3]^=v[11];
    v[4]^=v[12];v[5]^=v[13];v[6]^=v[14];v[7]^=v[15];
    return array<u32,8>(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
}

// ── PowHash: blake3 of 80 bytes in two blocks ─────────────────────────────────
// Block1 (64 B, CHUNK_START=1): pre_pow_hash[32] || timestamp[8] || zeros[24]
// Block2 (16 B, CHUNK_END|ROOT=10): zeros[8] || nonce[8]
fn pow_hash(ph:array<u32,8>,ts_lo:u32,ts_hi:u32,nonce_lo:u32,nonce_hi:u32)->array<u32,8>{
    var m1:array<u32,16>;
    m1[0]=ph[0];m1[1]=ph[1];m1[2]=ph[2];m1[3]=ph[3];
    m1[4]=ph[4];m1[5]=ph[5];m1[6]=ph[6];m1[7]=ph[7];
    m1[8]=ts_lo;m1[9]=ts_hi;
    // m1[10..15] = 0 (zeros[24])
    m1[10]=0u;m1[11]=0u;m1[12]=0u;m1[13]=0u;m1[14]=0u;m1[15]=0u;
    let cv=b3_compress(IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7,&m1,0u,0u,64u,1u);
    var m2:array<u32,16>;
    // m2[0..1] = zeros[8], m2[2..3] = nonce
    m2[0]=0u;m2[1]=0u;m2[2]=nonce_lo;m2[3]=nonce_hi;
    m2[4]=0u;m2[5]=0u;m2[6]=0u;m2[7]=0u;
    m2[8]=0u;m2[9]=0u;m2[10]=0u;m2[11]=0u;
    m2[12]=0u;m2[13]=0u;m2[14]=0u;m2[15]=0u;
    return b3_compress(cv[0],cv[1],cv[2],cv[3],cv[4],cv[5],cv[6],cv[7],&m2,0u,0u,16u,10u);
}

// ── KHeavyHash final: blake3 of 32 bytes (single block, flags=11) ─────────────
fn kheavy_final(d:array<u32,8>)->array<u32,8>{
    var m:array<u32,16>;
    m[0]=d[0];m[1]=d[1];m[2]=d[2];m[3]=d[3];
    m[4]=d[4];m[5]=d[5];m[6]=d[6];m[7]=d[7];
    m[8]=0u;m[9]=0u;m[10]=0u;m[11]=0u;m[12]=0u;m[13]=0u;m[14]=0u;m[15]=0u;
    return b3_compress(IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7,&m,0u,0u,32u,11u);
}

// ── 256-bit LE ≤ comparison ────────────────────────────────────────────────────
fn le256(a:array<u32,8>,b:array<u32,8>)->bool{
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

// ── Bindings ───────────────────────────────────────────────────────────────────
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
    dbg_hash: array<u32,8>,  // GPU final hash of winning nonce (for CPU comparison)
}

@group(0) @binding(0) var<storage,read>        kparams: KHeavyParams;
// matrix: 4096 u32s (matrix[row*64+col], each value 0-15)
@group(0) @binding(1) var<storage,read>        mat:     array<u32>;
@group(0) @binding(2) var<storage,read_write>  kout:    KHeavyOutput;

// ── Kernel ─────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let delta=gid.x;
    var nonce_lo=kparams.nonce_base_lo+delta;
    var nonce_hi=kparams.nonce_base_hi;
    if nonce_lo<delta{nonce_hi+=1u;}

    // Step 1: PowHash (80-byte blake3)
    // Copy into var so dynamic indexing is valid (naga forbids indexing let-bound arrays)
    var h1: array<u32,8>;
    {let t=pow_hash(kparams.pre_pow_hash,kparams.timestamp_lo,kparams.timestamp_hi,nonce_lo,nonce_hi);
     h1[0]=t[0];h1[1]=t[1];h1[2]=t[2];h1[3]=t[3];
     h1[4]=t[4];h1[5]=t[5];h1[6]=t[6];h1[7]=t[7];}

    // Step 2a: expand h1 (32 bytes = 8 u32) into 64 nibbles
    // h1[word] packs bytes LE: byte0=bits[7:0], byte1=bits[15:8], …
    var nibbles: array<u32,64>;
    for(var w:u32=0u;w<8u;w++){
        let word=h1[w];
        let b0=word&0xFFu;
        let b1=(word>>8u)&0xFFu;
        let b2=(word>>16u)&0xFFu;
        let b3=(word>>24u)&0xFFu;
        nibbles[w*8u+0u]=(b0>>4u)&0xFu; nibbles[w*8u+1u]=b0&0xFu;
        nibbles[w*8u+2u]=(b1>>4u)&0xFu; nibbles[w*8u+3u]=b1&0xFu;
        nibbles[w*8u+4u]=(b2>>4u)&0xFu; nibbles[w*8u+5u]=b2&0xFu;
        nibbles[w*8u+6u]=(b3>>4u)&0xFu; nibbles[w*8u+7u]=b3&0xFu;
    }

    // Step 2b: matrix-vector multiply, fold nibbles, XOR with h1 bytes, pack to 8 u32
    var prod: array<u32,8>;
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
    var prod_final: array<u32,8>;
    {let t=kheavy_final(prod);
     prod_final[0]=t[0];prod_final[1]=t[1];prod_final[2]=t[2];prod_final[3]=t[3];
     prod_final[4]=t[4];prod_final[5]=t[5];prod_final[6]=t[6];prod_final[7]=t[7];}
    let final_hash=prod_final;

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
