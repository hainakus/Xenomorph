use revm::primitives::{Address, Bytes, B256, U256};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use thiserror::Error;

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum TxError {
    #[error("RLP decode: {0}")]
    Rlp(String),
    #[error("signature recovery: {0}")]
    Signature(String),
    #[error("unsupported tx type 0x{0:02x}")]
    UnsupportedType(u8),
    #[error("invalid address length {0}")]
    AddressLen(usize),
}

// ── Keccak helper ─────────────────────────────────────────────────────────────

pub fn keccak256(data: &[u8]) -> [u8; 32] {
    Keccak256::digest(data).into()
}

// ── Decoded transaction ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DecodedTx {
    pub hash: B256,
    pub from: Address,
    pub to: Option<Address>,
    pub nonce: u64,
    pub gas_price: U256,
    pub gas_limit: u64,
    pub value: U256,
    pub data: Bytes,
    pub chain_id: u64,
}

// ── Log ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmLog {
    #[serde(rename = "address")]
    pub address: String,           // "0x{20-byte hex}"
    #[serde(rename = "topics")]
    pub topics: Vec<String>,       // ["0x{32-byte hex}", ...]
    #[serde(rename = "data")]
    pub data: String,              // "0x{hex}"
    #[serde(rename = "blockNumber")]
    pub block_number: String,
    #[serde(rename = "transactionHash")]
    pub transaction_hash: String,
    #[serde(rename = "transactionIndex")]
    pub transaction_index: String,
    #[serde(rename = "blockHash")]
    pub block_hash: String,
    #[serde(rename = "logIndex")]
    pub log_index: String,
    #[serde(rename = "removed")]
    pub removed: bool,
}

// ── Receipt ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmReceipt {
    #[serde(rename = "transactionHash")]
    pub transaction_hash: String,
    #[serde(rename = "blockHash")]
    pub block_hash: String,           // "0x0..." until patched in mine_block
    #[serde(rename = "blockNumber")]
    pub block_number: String,
    #[serde(rename = "from")]
    pub from: String,
    #[serde(rename = "to")]
    pub to: Option<String>,
    #[serde(rename = "contractAddress")]
    pub contract_address: Option<String>,
    #[serde(rename = "transactionIndex")]
    pub transaction_index: String,
    #[serde(rename = "gasUsed")]
    pub gas_used: String,
    #[serde(rename = "cumulativeGasUsed")]
    pub cumulative_gas_used: String,  // patched in mine_block
    #[serde(rename = "effectiveGasPrice")]
    pub effective_gas_price: String,
    #[serde(rename = "status")]
    pub status: String,
    #[serde(rename = "logs")]
    pub logs: Vec<EvmLog>,
    #[serde(rename = "logsBloom")]
    pub logs_bloom: String,
    #[serde(rename = "type")]
    pub tx_type: String,
}

// ── Raw transaction decoder (legacy / EIP-155) ────────────────────────────────

pub fn decode_raw_tx(raw: &[u8]) -> Result<DecodedTx, TxError> {
    if raw.is_empty() {
        return Err(TxError::Rlp("empty tx".into()));
    }
    match raw[0] {
        0x01 => decode_eip2930(&raw[1..]), // EIP-2930: chain_id, nonce, gas_price, gas_limit, to, value, data, access_list, y, r, s
        0x02 => decode_eip1559(&raw[1..]), // EIP-1559: chain_id, nonce, max_priority_fee, max_fee, gas, to, value, data, access_list, y, r, s
        b if b >= 0xc0 => decode_legacy(raw), // RLP list — legacy tx
        b => Err(TxError::UnsupportedType(b)),
    }
}

fn decode_eip2930(body: &[u8]) -> Result<DecodedTx, TxError> {
    use rlp::Rlp;
    let rlp = Rlp::new(body);

    let chain_id: u64    = rlp.val_at(0).map_err(|e| TxError::Rlp(e.to_string()))?;
    let nonce: u64       = rlp.val_at(1).map_err(|e| TxError::Rlp(e.to_string()))?;
    let gp_bytes: Vec<u8>= rlp.val_at(2).map_err(|e| TxError::Rlp(e.to_string()))?;
    let gas_limit: u64   = rlp.val_at(3).map_err(|e| TxError::Rlp(e.to_string()))?;
    let to_bytes: Vec<u8>= rlp.val_at(4).map_err(|e| TxError::Rlp(e.to_string()))?;
    let val_bytes: Vec<u8>= rlp.val_at(5).map_err(|e| TxError::Rlp(e.to_string()))?;
    let data: Vec<u8>    = rlp.val_at(6).map_err(|e| TxError::Rlp(e.to_string()))?;
    // field 7 = access_list (skip)
    let y_parity: u8     = rlp.val_at(8).map_err(|e| TxError::Rlp(e.to_string()))?;
    let r_bytes: Vec<u8> = rlp.val_at(9).map_err(|e| TxError::Rlp(e.to_string()))?;
    let s_bytes: Vec<u8> = rlp.val_at(10).map_err(|e| TxError::Rlp(e.to_string()))?;

    // Signing hash = keccak256(0x01 || RLP([chain_id, nonce, gas_price, gas_limit, to, value, data, access_list]))
    let access_list_raw: Vec<u8> = rlp.at(7).map_err(|e| TxError::Rlp(e.to_string()))?.as_raw().to_vec();
    let sign_body = build_eip2930_signing_rlp(chain_id, &gp_bytes, nonce, gas_limit, &to_bytes, &val_bytes, &data, &access_list_raw);
    let mut prefixed = vec![0x01u8];
    prefixed.extend_from_slice(&sign_body);
    let signing_hash = keccak256(&prefixed);

    let from_raw = recover_signer(&signing_hash, y_parity & 1, &r_bytes, &s_bytes)?;
    let from = Address::from_slice(&from_raw);
    let to = parse_to(&to_bytes)?;
    let gas_price = bytes_to_u256(&gp_bytes);
    let value = bytes_to_u256(&val_bytes);

    let mut full_raw = vec![0x01u8];
    full_raw.extend_from_slice(body);
    let hash = B256::from_slice(&keccak256(&full_raw));

    Ok(DecodedTx { hash, from, to, nonce, gas_price, gas_limit, value, data: data.into(), chain_id })
}

fn decode_eip1559(body: &[u8]) -> Result<DecodedTx, TxError> {
    use rlp::Rlp;
    let rlp = Rlp::new(body);

    let chain_id: u64       = rlp.val_at(0).map_err(|e| TxError::Rlp(e.to_string()))?;
    let nonce: u64          = rlp.val_at(1).map_err(|e| TxError::Rlp(e.to_string()))?;
    let _mpfpg: Vec<u8>     = rlp.val_at(2).map_err(|e| TxError::Rlp(e.to_string()))?; // max_priority_fee
    let max_fee: Vec<u8>    = rlp.val_at(3).map_err(|e| TxError::Rlp(e.to_string()))?; // max_fee_per_gas → gas_price
    let gas_limit: u64      = rlp.val_at(4).map_err(|e| TxError::Rlp(e.to_string()))?;
    let to_bytes: Vec<u8>   = rlp.val_at(5).map_err(|e| TxError::Rlp(e.to_string()))?;
    let val_bytes: Vec<u8>  = rlp.val_at(6).map_err(|e| TxError::Rlp(e.to_string()))?;
    let data: Vec<u8>       = rlp.val_at(7).map_err(|e| TxError::Rlp(e.to_string()))?;
    // field 8 = access_list (skip)
    let y_parity: u8        = rlp.val_at(9).map_err(|e| TxError::Rlp(e.to_string()))?;
    let r_bytes: Vec<u8>    = rlp.val_at(10).map_err(|e| TxError::Rlp(e.to_string()))?;
    let s_bytes: Vec<u8>    = rlp.val_at(11).map_err(|e| TxError::Rlp(e.to_string()))?;

    // Signing hash = keccak256(0x02 || RLP([chain_id, nonce, max_priority_fee, max_fee, gas, to, value, data, access_list]))
    let mpfpg_bytes: Vec<u8> = rlp.val_at(2).map_err(|e| TxError::Rlp(e.to_string()))?;
    let access_list_raw: Vec<u8> = rlp.at(8).map_err(|e| TxError::Rlp(e.to_string()))?.as_raw().to_vec();
    let sign_body = build_eip1559_signing_rlp(chain_id, nonce, &mpfpg_bytes, &max_fee, gas_limit, &to_bytes, &val_bytes, &data, &access_list_raw);
    let mut prefixed = vec![0x02u8];
    prefixed.extend_from_slice(&sign_body);
    let signing_hash = keccak256(&prefixed);

    let from_raw = recover_signer(&signing_hash, y_parity & 1, &r_bytes, &s_bytes)?;
    let from = Address::from_slice(&from_raw);
    let to = parse_to(&to_bytes)?;
    let gas_price = bytes_to_u256(&max_fee); // use max_fee as effective gas price
    let value = bytes_to_u256(&val_bytes);

    let mut full_raw = vec![0x02u8];
    full_raw.extend_from_slice(body);
    let hash = B256::from_slice(&keccak256(&full_raw));

    Ok(DecodedTx { hash, from, to, nonce, gas_price, gas_limit, value, data: data.into(), chain_id })
}

fn parse_to(to_bytes: &[u8]) -> Result<Option<Address>, TxError> {
    if to_bytes.is_empty() { Ok(None) }
    else if to_bytes.len() == 20 { Ok(Some(Address::from_slice(to_bytes))) }
    else { Err(TxError::AddressLen(to_bytes.len())) }
}

fn build_eip2930_signing_rlp(chain_id: u64, gp: &[u8], nonce: u64, gas_limit: u64, to: &[u8], value: &[u8], data: &[u8], access_list_raw: &[u8]) -> Vec<u8> {
    use rlp::RlpStream;
    let mut s = RlpStream::new_list(8);
    s.append(&chain_id);
    s.append(&nonce);
    s.append(&strip_leading_zeros(gp));
    s.append(&gas_limit);
    if to.is_empty() { s.append_empty_data(); } else { s.append(&to); }
    s.append(&strip_leading_zeros(value));
    s.append(&data);
    s.append_raw(access_list_raw, 1);
    s.out().to_vec()
}

fn build_eip1559_signing_rlp(chain_id: u64, nonce: u64, mpfpg: &[u8], max_fee: &[u8], gas_limit: u64, to: &[u8], value: &[u8], data: &[u8], access_list_raw: &[u8]) -> Vec<u8> {
    use rlp::RlpStream;
    let mut s = RlpStream::new_list(9);
    s.append(&chain_id);
    s.append(&nonce);
    s.append(&strip_leading_zeros(mpfpg));
    s.append(&strip_leading_zeros(max_fee));
    s.append(&gas_limit);
    if to.is_empty() { s.append_empty_data(); } else { s.append(&to); }
    s.append(&strip_leading_zeros(value));
    s.append(&data);
    s.append_raw(access_list_raw, 1);
    s.out().to_vec()
}

fn decode_legacy(raw: &[u8]) -> Result<DecodedTx, TxError> {
    use rlp::Rlp;
    let rlp = Rlp::new(raw);

    let nonce: u64 = rlp.val_at(0).map_err(|e| TxError::Rlp(e.to_string()))?;
    let gas_price_bytes: Vec<u8> = rlp.val_at(1).map_err(|e| TxError::Rlp(e.to_string()))?;
    let gas_limit: u64 = rlp.val_at(2).map_err(|e| TxError::Rlp(e.to_string()))?;
    let to_bytes: Vec<u8> = rlp.val_at(3).map_err(|e| TxError::Rlp(e.to_string()))?;
    let value_bytes: Vec<u8> = rlp.val_at(4).map_err(|e| TxError::Rlp(e.to_string()))?;
    let data: Vec<u8> = rlp.val_at(5).map_err(|e| TxError::Rlp(e.to_string()))?;
    let v: u64 = rlp.val_at(6).map_err(|e| TxError::Rlp(e.to_string()))?;
    let r_bytes: Vec<u8> = rlp.val_at(7).map_err(|e| TxError::Rlp(e.to_string()))?;
    let s_bytes: Vec<u8> = rlp.val_at(8).map_err(|e| TxError::Rlp(e.to_string()))?;

    let gas_price = bytes_to_u256(&gas_price_bytes);
    let value = bytes_to_u256(&value_bytes);

    let to = if to_bytes.is_empty() {
        None
    } else if to_bytes.len() == 20 {
        Some(Address::from_slice(&to_bytes))
    } else {
        return Err(TxError::AddressLen(to_bytes.len()));
    };

    // EIP-155: chain_id = (v - 35) / 2  or  (v - 36) / 2
    let (chain_id, rec_id) = if v >= 35 {
        ((v - 35) / 2, ((v - 35) % 2) as u8)
    } else {
        (1u64, (v - 27) as u8)
    };

    // Signing hash (EIP-155)
    let signing_data = build_signing_rlp(nonce, &gas_price_bytes, gas_limit, &to_bytes, &value_bytes, &data, chain_id);
    let signing_hash = keccak256(&signing_data);

    let from_raw = recover_signer(&signing_hash, rec_id, &r_bytes, &s_bytes)?;
    let from = Address::from_slice(&from_raw);

    let hash_bytes = keccak256(raw);
    let hash = B256::from_slice(&hash_bytes);

    Ok(DecodedTx { hash, from, to, nonce, gas_price, gas_limit, value, data: data.into(), chain_id })
}

fn bytes_to_u256(b: &[u8]) -> U256 {
    if b.is_empty() { U256::ZERO } else { U256::from_be_slice(b) }
}

fn strip_leading_zeros(b: &[u8]) -> &[u8] {
    let start = b.iter().position(|&x| x != 0).unwrap_or(b.len());
    &b[start..]
}

fn build_signing_rlp(
    nonce: u64, gas_price: &[u8], gas_limit: u64,
    to: &[u8], value: &[u8], data: &[u8], chain_id: u64,
) -> Vec<u8> {
    use rlp::RlpStream;
    let mut s = RlpStream::new_list(9);
    s.append(&nonce);
    s.append(&strip_leading_zeros(gas_price));
    s.append(&gas_limit);
    if to.is_empty() {
        s.append_empty_data();
    } else {
        s.append(&to);
    }
    s.append(&strip_leading_zeros(value));
    s.append(&data);
    s.append(&chain_id);
    s.append(&0u64);
    s.append(&0u64);
    s.out().to_vec()
}

fn recover_signer(hash: &[u8; 32], rec_id: u8, r: &[u8], s: &[u8]) -> Result<[u8; 20], TxError> {
    use secp256k1::{
        ecdsa::{RecoverableSignature, RecoveryId},
        Message, Secp256k1,
    };

    let secp = Secp256k1::new();
    let msg = Message::from_digest_slice(hash).map_err(|e| TxError::Signature(e.to_string()))?;
    let rec = RecoveryId::from_i32(rec_id as i32).map_err(|e| TxError::Signature(e.to_string()))?;

    let mut rs = [0u8; 64];
    let pad = |src: &[u8], dst: &mut [u8; 32]| {
        let start = 32usize.saturating_sub(src.len());
        dst[start..].copy_from_slice(&src[src.len().saturating_sub(32)..]);
    };
    let mut r32 = [0u8; 32];
    let mut s32 = [0u8; 32];
    pad(r, &mut r32);
    pad(s, &mut s32);
    rs[..32].copy_from_slice(&r32);
    rs[32..].copy_from_slice(&s32);

    let sig = RecoverableSignature::from_compact(&rs, rec).map_err(|e| TxError::Signature(e.to_string()))?;
    let pubkey = secp.recover_ecdsa(&msg, &sig).map_err(|e| TxError::Signature(e.to_string()))?;

    let pub_bytes = pubkey.serialize_uncompressed();
    let addr_hash = keccak256(&pub_bytes[1..]);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&addr_hash[12..]);
    Ok(addr)
}
