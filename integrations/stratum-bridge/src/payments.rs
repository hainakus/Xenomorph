use std::{fmt, sync::Arc};

use anyhow::{bail, Context, Result};
use kaspa_addresses::Address;
use kaspa_consensus_core::{
    hashing::sighash::{calc_schnorr_signature_hash, SigHashReusedValues},
    hashing::sighash_type::SIG_HASH_ALL,
    subnets::SUBNETWORK_ID_NATIVE,
    tx::{
        MutableTransaction, Transaction, TransactionInput, TransactionOutpoint, TransactionOutput,
        UtxoEntry,
    },
};
use kaspa_core::{info, warn};
use kaspa_grpc_client::GrpcClient;
use kaspa_rpc_core::{
    api::rpc::RpcApi, RpcAddress, RpcTransaction, RpcTransactionInput, RpcTransactionOutput,
    RpcTransactionOutpoint, RpcTransactionId,
};
use kaspa_txscript::pay_to_address_script;
use secp256k1::{Keypair, Message};

use crate::accounting::PendingPayout;

/// Signals that a payout failure is **transient** — the block stays `confirmed`
/// and will be retried on the next monitor cycle.
#[derive(Debug)]
pub struct RetryablePayoutError(pub String);
impl fmt::Display for RetryablePayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for RetryablePayoutError {}

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct PaymentConfig {
    /// How many DAA-score steps after the mined block before paying out.
    pub confirm_depth:    u64,
    /// Minimum amount to send per miner per payout (sompi).
    /// Outputs below this threshold are skipped and rolled over to the next block.
    pub min_payout_sompi: u64,
    /// Pool operator fee as a percentage (e.g. 1.0 = 1 %).
    pub pool_fee_percent: f64,
    /// Estimated transaction fee per output (sompi). Used for UTXO selection.
    pub fee_per_output:   u64,
}

impl Default for PaymentConfig {
    fn default() -> Self {
        Self {
            confirm_depth:    1000,
            min_payout_sompi: 100_000, // 0.001 XENOM
            pool_fee_percent: 1.0,
            fee_per_output:   2_000,
        }
    }
}

// ── Payout execution ──────────────────────────────────────────────────────────

/// Build, sign and broadcast a payout transaction for one confirmed block.
///
/// Requires the Xenom node to be started with `--utxoindex`.
pub async fn execute_payout(
    rpc:          &Arc<GrpcClient>,
    pool_address: &RpcAddress,
    keypair:      &Keypair,
    payout:       &PendingPayout,
    cfg:          &PaymentConfig,
) -> Result<RpcTransactionId> {
    // ── 1. Fetch all pool UTXOs ───────────────────────────────────────────────
    let utxo_entries = rpc
        .get_utxos_by_addresses(vec![pool_address.clone()])
        .await
        .context("get_utxos_by_addresses (node must be started with --utxoindex)")?;

    if utxo_entries.is_empty() {
        return Err(anyhow::Error::new(RetryablePayoutError(
            format!("No UTXOs available for pool address {pool_address} — will retry next cycle")
        )));
    }

    // ── 2. Calculate funds and outputs ───────────────────────────────────────
    let total_available: u64 = utxo_entries.iter().map(|e| e.utxo_entry.amount).sum();

    // Apply pool fee
    let after_fee = (total_available as f64 * (1.0 - cfg.pool_fee_percent / 100.0)) as u64;

    // Plan per-miner outputs; skip miners below minimum payout or with no valid address
    let mut invalid_addr_count = 0usize;
    let mut below_min_count    = 0usize;
    let outputs_plan: Vec<(RpcAddress, u64)> = payout
        .proportions
        .iter()
        .filter_map(|(worker, proportion)| {
            let addr_str = worker.split('.').next().unwrap_or(worker);
            let addr: RpcAddress = match Address::try_from(addr_str) {
                Ok(a)  => a,
                Err(_) => {
                    warn!(
                        "Skipping payout to '{worker}': '{addr_str}' is not a valid xenom: address. \
                         Miners must connect with username  xenom:qYOURADDR[.workername]"
                    );
                    invalid_addr_count += 1;
                    return None;
                }
            };
            let amount = (after_fee as f64 * proportion) as u64;
            if amount < cfg.min_payout_sompi {
                warn!("Skipping payout to {worker}: {amount} sompi < min {}", cfg.min_payout_sompi);
                below_min_count += 1;
                return None;
            }
            Some((addr, amount))
        })
        .collect();

    if outputs_plan.is_empty() {
        let msg = if invalid_addr_count > 0 && below_min_count == 0 {
            format!(
                "No payable miners: {invalid_addr_count} worker(s) have no valid xenom: address — \
                 miners must connect with username  xenom:qYOURADDR[.workername]  (will retry)"
            )
        } else {
            format!(
                "No miner payouts above minimum threshold ({} sompi): \
                 {invalid_addr_count} invalid address(es), {below_min_count} below minimum (will retry)",
                cfg.min_payout_sompi
            )
        };
        return Err(anyhow::Error::new(RetryablePayoutError(msg)));
    }

    let total_out: u64  = outputs_plan.iter().map(|(_, v)| *v).sum();
    let fee_total:  u64 = cfg.fee_per_output * (outputs_plan.len() as u64 + 1); // +1 for change
    let need:       u64 = total_out + fee_total;

    // ── 3. Greedy UTXO selection ──────────────────────────────────────────────
    let mut selected       = Vec::new();
    let mut selected_total = 0u64;
    for entry in &utxo_entries {
        selected.push(entry);
        selected_total += entry.utxo_entry.amount;
        if selected_total >= need { break; }
    }
    if selected_total < need {
        return Err(anyhow::Error::new(RetryablePayoutError(
            format!("Insufficient pool funds: available {selected_total} sompi, need {need} sompi — will retry next cycle")
        )));
    }

    // ── 4. Build transaction outputs ─────────────────────────────────────────
    let mut tx_outputs: Vec<TransactionOutput> = outputs_plan
        .iter()
        .map(|(addr, amount)| TransactionOutput {
            value:             *amount,
            script_public_key: pay_to_address_script(addr),
        })
        .collect();

    // Change output back to the pool address
    let change = selected_total.saturating_sub(total_out + fee_total);
    if change >= cfg.min_payout_sompi {
        tx_outputs.push(TransactionOutput {
            value:             change,
            script_public_key: pay_to_address_script(pool_address),
        });
    }

    // ── 5. Build unsigned inputs ──────────────────────────────────────────────
    let tx_inputs: Vec<TransactionInput> = selected
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

    let tx = Transaction::new(0, tx_inputs, tx_outputs, 0, SUBNETWORK_ID_NATIVE, 0, vec![]);

    // ── 6. Sign each input (P2PK Schnorr) ────────────────────────────────────
    let utxo_for_signing: Vec<UtxoEntry> = selected
        .iter()
        .map(|e| UtxoEntry {
            amount:           e.utxo_entry.amount,
            script_public_key: e.utxo_entry.script_public_key.clone(),
            block_daa_score:  e.utxo_entry.block_daa_score,
            is_coinbase:      e.utxo_entry.is_coinbase,
        })
        .collect();

    let mut mutable_tx     = MutableTransaction::with_entries(tx, utxo_for_signing);
    let mut reused_values  = SigHashReusedValues::new();
    let n_inputs           = mutable_tx.tx.inputs.len();

    for i in 0..n_inputs {
        let sig_hash = calc_schnorr_signature_hash(
            &mutable_tx.as_verifiable(), i, SIG_HASH_ALL, &mut reused_values,
        );
        let msg = Message::from_digest_slice(sig_hash.as_bytes().as_slice())
            .context("secp256k1 message from sig_hash")?;
        let sig = keypair.sign_schnorr(msg);

        // P2PK Schnorr sig_script: OP_DATA_65 (0x41) || sig[64] || SIG_HASH_ALL
        let mut sig_script = Vec::with_capacity(66);
        sig_script.push(0x41u8); // OP_DATA_65
        sig_script.extend_from_slice(sig.as_ref());
        sig_script.push(SIG_HASH_ALL.to_u8());
        mutable_tx.tx.inputs[i].signature_script = sig_script;
    }

    // ── 7. Convert and submit ─────────────────────────────────────────────────
    let rpc_tx = consensus_tx_to_rpc(mutable_tx.tx);
    let tx_id  = rpc
        .submit_transaction(rpc_tx, false)
        .await
        .context("submit_transaction")?;

    info!(
        "Payout tx submitted: {tx_id}  outputs={}  total_out={} sompi",
        outputs_plan.len(),
        total_out
    );
    Ok(tx_id)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

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
