use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Context, Result};
use kaspa_addresses::{Address, Prefix, Version};
use kaspa_consensus_core::{
    hashing::sighash::{calc_schnorr_signature_hash, SigHashReusedValues},
    hashing::sighash_type::SIG_HASH_ALL,
    subnets::SUBNETWORK_ID_NATIVE,
    tx::{
        MutableTransaction, Transaction, TransactionInput, TransactionOutpoint, TransactionOutput,
        UtxoEntry,
    },
};
use kaspa_grpc_client::GrpcClient;
use kaspa_rpc_core::{
    api::rpc::RpcApi, RpcAddress, RpcTransaction, RpcTransactionInput, RpcTransactionOutput,
    RpcTransactionOutpoint,
};
use kaspa_txscript::pay_to_address_script;
use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Coinbase outputs must mature this many DAA-score steps before they can be spent.
const COINBASE_MATURITY: u64 = 100;

/// Default relay fee per input (sompi).  Keeps mass well under 100_000 node limit.
pub const DEFAULT_FEE_PER_INPUT: u64 = 2_000;

// ── Public API ────────────────────────────────────────────────────────────────

/// Build, sign and submit a minimal anchor transaction.
///
/// The `anchor_bytes` are embedded verbatim in `tx.payload` — the Kaspa/Xenom
/// native data field.  No OP_RETURN output is needed.
///
/// Transaction structure:
/// - Inputs:  all mature UTXOs held by the derived funding address
/// - Outputs: 1 change output back to the funding address (value = total_in - fee)
/// - Payload: `anchor_bytes`
///
/// Returns the transaction ID hex string on success.
pub async fn submit_anchor(
    rpc:          &Arc<GrpcClient>,
    keypair:      &Keypair,
    anchor_bytes: &[u8],
    fee_per_input: u64,
    prefix:        Prefix,
) -> Result<String> {
    let (pubkey, _) = keypair.x_only_public_key();
    let address     = Address::new(prefix, Version::PubKey, &pubkey.serialize());
    let rpc_address: RpcAddress = address.clone();

    // ── 1. Fetch mature UTXOs ─────────────────────────────────────────────────
    let current_daa = rpc
        .get_block_dag_info()
        .await
        .context("get_block_dag_info RPC failed")?
        .virtual_daa_score;

    let utxo_entries: Vec<_> = rpc
        .get_utxos_by_addresses(vec![rpc_address.clone()])
        .await
        .context("get_utxos_by_addresses RPC failed — node needs --utxoindex?")?
        .into_iter()
        .filter(|e| {
            !e.utxo_entry.is_coinbase
                || current_daa.saturating_sub(e.utxo_entry.block_daa_score) >= COINBASE_MATURITY
        })
        .collect();

    if utxo_entries.is_empty() {
        anyhow::bail!(
            "No mature UTXOs for funding address {rpc_address} — \
             fund the address before submitting an anchor tx"
        );
    }

    // ── 2. Compute fee and change ─────────────────────────────────────────────
    let n_inputs: u64 = utxo_entries.len() as u64;
    let total_in: u64 = utxo_entries.iter().map(|e| e.utxo_entry.amount).sum();
    let fee          = fee_per_input * n_inputs;

    if total_in <= fee {
        anyhow::bail!(
            "Insufficient funds: total UTXOs = {total_in} sompi, fee = {fee} sompi. \
             Fund the address {rpc_address} with more XENOM."
        );
    }

    let change = total_in - fee;

    // ── 3. Build inputs ───────────────────────────────────────────────────────
    let tx_inputs: Vec<TransactionInput> = utxo_entries
        .iter()
        .map(|e| TransactionInput {
            previous_outpoint: TransactionOutpoint {
                transaction_id: e.outpoint.transaction_id,
                index:          e.outpoint.index,
            },
            signature_script: vec![],
            sequence:         0,
            sig_op_count:     1,
        })
        .collect();

    // ── 4. Build outputs (single change output) ───────────────────────────────
    let tx_outputs = vec![TransactionOutput {
        value:             change,
        script_public_key: pay_to_address_script(&rpc_address),
    }];

    // ── 5. Build transaction — embed anchor_bytes in tx.payload ───────────────
    let tx = Transaction::new(
        0,
        tx_inputs,
        tx_outputs,
        0,
        SUBNETWORK_ID_NATIVE,
        0,
        anchor_bytes.to_vec(),
    );

    // ── 6. Sign all inputs (P2PK Schnorr) ────────────────────────────────────
    let utxo_for_signing: Vec<UtxoEntry> = utxo_entries
        .iter()
        .map(|e| UtxoEntry {
            amount:            e.utxo_entry.amount,
            script_public_key: e.utxo_entry.script_public_key.clone(),
            block_daa_score:   e.utxo_entry.block_daa_score,
            is_coinbase:       e.utxo_entry.is_coinbase,
        })
        .collect();

    let mut mutable_tx    = MutableTransaction::with_entries(tx, utxo_for_signing);
    let mut reused_values = SigHashReusedValues::new();
    let n                 = mutable_tx.tx.inputs.len();

    for i in 0..n {
        let sig_hash = calc_schnorr_signature_hash(
            &mutable_tx.as_verifiable(), i, SIG_HASH_ALL, &mut reused_values,
        );
        let msg = Message::from_digest_slice(sig_hash.as_bytes().as_slice())
            .context("secp256k1 message from sig_hash")?;
        let sig = keypair.sign_schnorr(msg);

        let mut sig_script = Vec::with_capacity(66);
        sig_script.push(0x41u8);             // OP_DATA_65
        sig_script.extend_from_slice(sig.as_ref());
        sig_script.push(SIG_HASH_ALL.to_u8());
        mutable_tx.tx.inputs[i].signature_script = sig_script;
    }

    // ── 7. Submit ─────────────────────────────────────────────────────────────
    let rpc_tx  = consensus_tx_to_rpc(mutable_tx.tx);
    let tx_id   = rpc
        .submit_transaction(rpc_tx, false)
        .await
        .context("submit_transaction RPC failed")?;

    log::info!(
        "Anchor tx submitted: {tx_id}  inputs={n_inputs}  payload={} bytes  change={change} sompi  fee={fee} sompi",
        anchor_bytes.len(),
    );

    Ok(tx_id.to_string())
}

// ── Keypair helper ────────────────────────────────────────────────────────────

/// Parse a 64-char hex secp256k1 private key into a `Keypair`.
pub fn keypair_from_hex(privkey_hex: &str) -> Result<Keypair> {
    let secp   = Secp256k1::new();
    let secret = SecretKey::from_str(privkey_hex)
        .context("invalid private key — expected 64 hex chars (32 bytes)")?;
    Ok(Keypair::from_secret_key(&secp, &secret))
}

/// Derive the Xenom address (P2PK) from a `Keypair` for the given network prefix.
pub fn address_from_keypair(keypair: &Keypair, prefix: Prefix) -> String {
    address_from_keypair_prefixed(keypair, prefix)
}

/// Derive address with an explicit network prefix (Mainnet / Devnet / Testnet).
pub fn address_from_keypair_prefixed(keypair: &Keypair, prefix: Prefix) -> String {
    let (pubkey, _) = keypair.x_only_public_key();
    let addr = Address::new(prefix, Version::PubKey, &pubkey.serialize());
    String::from(&addr)
}

// ── Internal helper ───────────────────────────────────────────────────────────

fn consensus_tx_to_rpc(tx: Transaction) -> RpcTransaction {
    RpcTransaction {
        version:       tx.version,
        inputs:        tx.inputs
            .into_iter()
            .map(|i| RpcTransactionInput {
                previous_outpoint: RpcTransactionOutpoint {
                    transaction_id: i.previous_outpoint.transaction_id,
                    index:          i.previous_outpoint.index,
                },
                signature_script: i.signature_script,
                sequence:         i.sequence,
                sig_op_count:     i.sig_op_count,
                verbose_data:     None,
            })
            .collect(),
        outputs:       tx.outputs
            .into_iter()
            .map(|o| RpcTransactionOutput {
                value:             o.value,
                script_public_key: o.script_public_key,
                verbose_data:      None,
            })
            .collect(),
        lock_time:     tx.lock_time,
        subnetwork_id: tx.subnetwork_id,
        gas:           tx.gas,
        payload:       tx.payload,
        mass:          0,
        verbose_data:  None,
    }
}
