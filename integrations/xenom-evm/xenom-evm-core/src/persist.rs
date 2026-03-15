/// Atomic JSON snapshot of the full EVM state.
///
/// Written after every `mine_block()` when `EvmChain` is created with a
/// `snapshot_path`.  On startup the node calls `load_snapshot()` to restore
/// balances, nonces, code, and storage from disk.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use revm::{
    db::InMemoryDB,
    primitives::{AccountInfo, Address, Bytecode, B256, U256},
};
use serde::{Deserialize, Serialize};

// ── On-disk format ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct StateSnapshot {
    pub chain_id: u64,
    pub block_number: u64,
    pub tx_index: u64,
    pub state_root: String,
    pub accounts: HashMap<String, AccountSnap>,
}

#[derive(Serialize, Deserialize)]
pub struct AccountSnap {
    pub balance: String,  // "0x{hex}"
    pub nonce: u64,
    pub code: String,     // "0x{hex}" — empty when EOA
    pub storage: HashMap<String, String>, // "0x{slot}" -> "0x{value}"
}

// ── Save ──────────────────────────────────────────────────────────────────────

pub fn save_snapshot(
    db: &InMemoryDB,
    chain_id: u64,
    block_number: u64,
    tx_index: u64,
    state_root: B256,
    path: &Path,
) -> std::io::Result<()> {
    let mut accounts = HashMap::new();

    for (addr, account) in &db.accounts {
        let info = &account.info;
        let code_hex = if let Some(code) = db.contracts.get(&info.code_hash) {
            format!("0x{}", hex::encode(code.original_bytes()))
        } else {
            "0x".to_string()
        };

        let storage: HashMap<String, String> = account
            .storage
            .iter()
            .filter(|(_, v)| **v != U256::ZERO)
            .map(|(k, v)| {
                (
                    format!("0x{}", hex::encode(k.to_be_bytes::<32>())),
                    format!("0x{}", hex::encode(v.to_be_bytes::<32>())),
                )
            })
            .collect();

        accounts.insert(
            format!("0x{}", hex::encode(addr)),
            AccountSnap {
                balance: format!("0x{}", hex::encode(info.balance.to_be_bytes::<32>())),
                nonce: info.nonce,
                code: code_hex,
                storage,
            },
        );
    }

    let snap = StateSnapshot {
        chain_id,
        block_number,
        tx_index,
        state_root: format!("0x{}", hex::encode(state_root)),
        accounts,
    };

    let json = serde_json::to_string_pretty(&snap)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // Atomic write: write to .tmp then rename
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, path)?;

    log::debug!("state snapshot saved to {}", path.display());
    Ok(())
}

// ── Load ──────────────────────────────────────────────────────────────────────

pub struct LoadedSnapshot {
    pub db: InMemoryDB,
    pub block_number: u64,
    pub tx_index: u64,
    pub state_root: B256,
}

pub fn load_snapshot(path: &Path) -> Option<LoadedSnapshot> {
    let data = std::fs::read(path).ok()?;
    let snap: StateSnapshot = serde_json::from_slice(&data)
        .map_err(|e| log::warn!("snapshot parse error: {e}"))
        .ok()?;

    let mut db = InMemoryDB::default();

    for (addr_hex, acc) in &snap.accounts {
        let addr_bytes = hex::decode(addr_hex.trim_start_matches("0x")).ok()?;
        let addr = Address::from_slice(&addr_bytes);

        let balance = parse_u256(&acc.balance).unwrap_or(U256::ZERO);
        let code_bytes = hex::decode(acc.code.trim_start_matches("0x")).unwrap_or_default();
        let has_code = !code_bytes.is_empty();
        let bytecode = if has_code {
            Bytecode::new_raw(code_bytes.into())
        } else {
            Bytecode::default()
        };

        let info = AccountInfo {
            balance,
            nonce: acc.nonce,
            code_hash: bytecode.hash_slow(),
            code: Some(bytecode.clone()),
        };
        db.insert_account_info(addr, info);
        if has_code {
            db.contracts.insert(bytecode.hash_slow(), bytecode);
        }

        for (slot_hex, val_hex) in &acc.storage {
            let slot = parse_u256(slot_hex).unwrap_or(U256::ZERO);
            let val = parse_u256(val_hex).unwrap_or(U256::ZERO);
            db.insert_account_storage(addr, slot, val).ok();
        }
    }

    let state_root = parse_b256(&snap.state_root).unwrap_or(B256::ZERO);
    log::info!(
        "state snapshot loaded: block={} accounts={} root={}",
        snap.block_number,
        snap.accounts.len(),
        snap.state_root
    );

    Some(LoadedSnapshot {
        db,
        block_number: snap.block_number,
        tx_index: snap.tx_index,
        state_root,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_u256(hex: &str) -> Option<U256> {
    let h = hex.trim_start_matches("0x");
    let padded = if h.len() % 2 == 1 { format!("0{h}") } else { h.to_string() };
    let bytes = hex::decode(&padded).ok()?;
    let mut arr = [0u8; 32];
    let start = 32usize.saturating_sub(bytes.len());
    arr[start..].copy_from_slice(&bytes[..bytes.len().min(32)]);
    Some(U256::from_be_bytes(arr))
}

fn parse_b256(hex: &str) -> Option<B256> {
    let bytes = hex::decode(hex.trim_start_matches("0x")).ok()?;
    if bytes.len() != 32 { return None; }
    Some(B256::from_slice(&bytes))
}

// ── Snapshot path helper ──────────────────────────────────────────────────────

pub fn snapshot_path(state_dir: &Path, chain_id: u64) -> PathBuf {
    state_dir.join(format!("evm-state-{chain_id}.json"))
}
