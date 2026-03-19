use std::{collections::HashSet, fmt, sync::Arc};

use anyhow::{Context, Result};
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
            min_payout_sompi: 20_000_000_000, // 200 XENOM
            pool_fee_percent: 1.0,
            fee_per_output:   2_000,
        }
    }
}

/// Coinbase outputs must wait this many DAA-score steps before they can be spent.
const COINBASE_MATURITY: u64 = 100;

// ── Mass estimation (KIP-9 Alpha: total = compute + storage) ─────────────────
/// C parameter: 100_000_000 (SOMPI_PER_XENOM) * 10_000 = 10^12
const STORAGE_MASS_PARAMETER: u64 = 100_000_000 * 10_000;
/// 5% headroom below the 100,000 node limit.
const MAX_SELECTABLE_MASS: u64 = 95_000;
/// Blank transaction bytes * mass_per_tx_byte(1) = 94
const BLANK_TX_MASS: u64 = 94;
/// Per signed P2PK input: 118 bytes * 1 + 1 sig_op * 1000 = 1118
const MASS_PER_INPUT: u64 = 1_118;
/// Per P2PK output: 52 bytes * 1 + (2+34) script bytes * 10 = 412
const MASS_PER_OUTPUT: u64 = 412;
/// Cap on inputs per transaction (keeps compute mass < 95k by itself).
const MAX_INPUTS: usize = 84;

/// Estimate storage mass using KIP-9 Alpha formula:
///   max(0, C·Σ(1/output_i) − C·N_inputs/mean_input)
fn estimate_storage_mass(input_amounts: &[u64], output_amounts: &[u64]) -> u64 {
    let n_ins = input_amounts.len() as u64;
    if n_ins == 0 {
        return u64::MAX;
    }
    let sum_ins: u64 = input_amounts.iter().sum();
    let mean_ins = (sum_ins / n_ins).max(1);
    let harmonic_outs: u64 = output_amounts
        .iter()
        .map(|&v| STORAGE_MASS_PARAMETER / v.max(1))
        .fold(0u64, |acc, x| acc.saturating_add(x));
    let arithmetic_ins = n_ins.saturating_mul(STORAGE_MASS_PARAMETER / mean_ins);
    harmonic_outs.saturating_sub(arithmetic_ins)
}

/// Estimate total transaction mass (compute + storage).
fn estimate_tx_mass(input_amounts: &[u64], output_amounts: &[u64]) -> u64 {
    let compute = BLANK_TX_MASS
        + input_amounts.len() as u64 * MASS_PER_INPUT
        + output_amounts.len() as u64 * MASS_PER_OUTPUT;
    let storage = estimate_storage_mass(input_amounts, output_amounts);
    compute.saturating_add(storage)
}

/// Minimum change amount to create a change output (avoid burning pool funds as TX fee).
/// Must be well above dust but much lower than min_payout_sompi.
const CHANGE_DUST_THRESHOLD: u64 = 100_000; // 0.001 XENOM

// ── Payout execution ──────────────────────────────────────────────────────────

/// Merge PPLNS proportions from multiple blocks into a single weighted-average
/// distribution.  Each block is weighted equally (one vote per block found).
/// Returns a sorted `Vec<(worker, merged_proportion)>`.
pub fn merge_proportions(payouts: &[PendingPayout]) -> Vec<(String, f64)> {
    let n = payouts.len() as f64;
    if n == 0.0 {
        return vec![];
    }
    let mut totals: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for payout in payouts {
        for (worker, prop) in &payout.proportions {
            *totals.entry(worker.clone()).or_default() += prop / n;
        }
    }
    let mut merged: Vec<(String, f64)> = totals.into_iter().collect();
    merged.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

/// Build, sign and broadcast ONE payout transaction covering all `proportions`.
///
/// Distributes ALL available pool UTXOs proportionally.
/// Call once with merged proportions from all confirmed blocks.
///
/// Requires the Xenom node to be started with `--utxoindex`.
///
/// `spent` is a per-cycle set of `(tx_id, index)` outpoints already submitted
/// earlier this cycle to avoid double-spend against unconfirmed mempool TXs.
pub async fn execute_payout(
    rpc:          &Arc<GrpcClient>,
    pool_address: &RpcAddress,
    keypair:      &Keypair,
    proportions:  &[(String, f64)],
    cfg:          &PaymentConfig,
    spent:        &mut HashSet<(RpcTransactionId, u32)>,
) -> Result<RpcTransactionId> {
    // ── 1. Fetch all pool UTXOs (mature only) ───────────────────────────────
    let current_daa = rpc
        .get_block_dag_info()
        .await
        .map_err(|e| anyhow::Error::new(RetryablePayoutError(
            format!("get_block_dag_info RPC failed: {e} — will retry")
        )))?
        .virtual_daa_score;

    let utxo_entries: Vec<_> = rpc
        .get_utxos_by_addresses(vec![pool_address.clone()])
        .await
        .map_err(|e| anyhow::Error::new(RetryablePayoutError(
            format!("get_utxos_by_addresses RPC failed (node needs --utxoindex?): {e} — will retry")
        )))?
        .into_iter()
        .filter(|e| {
            !e.utxo_entry.is_coinbase
                || current_daa.saturating_sub(e.utxo_entry.block_daa_score) >= COINBASE_MATURITY
        })
        .collect();

    if utxo_entries.is_empty() {
        return Err(anyhow::Error::new(RetryablePayoutError(
            format!("No mature UTXOs available for pool address {pool_address} — will retry next cycle")
        )));
    }

    // ── 2. UTXO pool: sort by value DESC (largest inputs → lowest storage mass) ─
    // Filter out outpoints already submitted in earlier payouts this cycle —
    // the node UTXO index does not see unconfirmed mempool spends.
    let mut sorted_utxos: Vec<_> = utxo_entries
        .iter()
        .filter(|e| !spent.contains(&(e.outpoint.transaction_id, e.outpoint.index)))
        .collect();
    sorted_utxos.sort_unstable_by(|a, b| b.utxo_entry.amount.cmp(&a.utxo_entry.amount));
    let total_pool: u64 = sorted_utxos.iter().map(|e| e.utxo_entry.amount).sum();

    if total_pool == 0 {
        return Err(anyhow::Error::new(RetryablePayoutError(
            format!("No spendable UTXOs for pool address {pool_address} — will retry next cycle")
        )));
    }

    // ── 3. Plan all payout outputs ───────────────────────────────────────────
    let after_fee = (total_pool as f64 * (1.0 - cfg.pool_fee_percent / 100.0)) as u64;

    let mut invalid_addr_count = 0usize;
    let mut below_min_count    = 0usize;
    let mut outputs_plan: Vec<(RpcAddress, u64)> = proportions
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

    // Sort outputs DESC so the largest (lowest storage-mass) outputs go first.
    outputs_plan.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    // ── 4a. Pre-filter outputs that can never satisfy the storage-mass limit ───
    // Storage mass for a single output v (best-case, with up to MAX_INPUTS inputs
    // each at the pool's current mean value):
    //   storage = C/v − C·N/mean_in
    // Viable when: C/v ≤ MAX_SELECTABLE_MASS + best_arithmetic_ins
    // => min_viable = C / (MAX_SELECTABLE_MASS + best_arithmetic_ins)
    {
        let n_best = sorted_utxos.len().min(MAX_INPUTS) as u64;
        if n_best > 0 {
            let sum_best: u64 = sorted_utxos.iter().take(n_best as usize)
                .map(|e| e.utxo_entry.amount).sum();
            let mean_best    = (sum_best / n_best).max(1);
            let best_arith   = n_best.saturating_mul(STORAGE_MASS_PARAMETER / mean_best);
            let max_harmonic = MAX_SELECTABLE_MASS.saturating_add(best_arith);
            let min_viable   = if max_harmonic == 0 { u64::MAX }
                               else { STORAGE_MASS_PARAMETER / max_harmonic };
            if min_viable > 1 {
                let before = outputs_plan.len();
                outputs_plan.retain(|(_, v)| *v >= min_viable);
                let dropped = before - outputs_plan.len();
                if dropped > 0 {
                    warn!(
                        "Dropped {dropped} payout output(s) below storage-mass minimum \
                         ({min_viable} sompi). Raise --min-payout-sompi to >= {min_viable}."
                    );
                }
            }
        }
    }

    if outputs_plan.is_empty() {
        return Err(anyhow::Error::new(RetryablePayoutError(
            "All payout outputs dropped by storage-mass filter — raise --min-payout-sompi (will retry)".into()
        )));
    }

    // ── 4. Batch loop: build/sign/submit one tx per mass-safe batch ───────────
    //
    // Storage mass formula (KIP-9 Alpha):
    //   storage = max(0, C·Σ(1/output_i) − C·N_inputs/mean_input)
    // Each batch uses a fresh, non-overlapping slice of sorted_utxos so that
    // no UTXO is double-spent across batches.  The change output of each batch
    // is sent back to the pool address and will be re-spendable in future cycles.
    let mut utxo_cursor  = 0usize;
    let mut out_cursor   = 0usize;
    let mut first_tx_id: Option<RpcTransactionId> = None;
    let mut batch_num    = 0u32;

    while out_cursor < outputs_plan.len() {
        batch_num += 1;

        // Greedily add outputs while estimated mass stays under the safe limit.
        let batch_start_utxo = utxo_cursor;
        let mut batch_inputs_end   = utxo_cursor;   // exclusive
        let mut batch_input_total  = 0u64;
        let mut batch_outputs: Vec<(RpcAddress, u64)> = vec![];

        'outer: for (addr, amount) in outputs_plan.iter().skip(out_cursor) {

            // Minimum inputs needed to cover this tentative batch.
            let tentative_n_out  = batch_outputs.len() + 1;
            let tentative_out_sum: u64 = batch_outputs.iter().map(|(_, v)| v).sum::<u64>() + amount;
            let tentative_fees   = cfg.fee_per_output * (tentative_n_out as u64 + 1); // +1 change
            let tentative_needed = tentative_out_sum + tentative_fees;

            // Expand input slice until we have enough funds, up to MAX_INPUTS.
            let mut temp_end   = batch_inputs_end;
            let mut temp_total = batch_input_total;
            while temp_total < tentative_needed {
                if temp_end >= sorted_utxos.len() || temp_end - batch_start_utxo >= MAX_INPUTS {
                    break 'outer; // ran out of UTXOs or hit input cap
                }
                temp_total += sorted_utxos[temp_end].utxo_entry.amount;
                temp_end   += 1;
            }

            // Estimate mass for this tentative batch (include change output).
            let tentative_change = temp_total.saturating_sub(tentative_needed);
            let input_amts: Vec<u64> = sorted_utxos[batch_start_utxo..temp_end]
                .iter().map(|e| e.utxo_entry.amount).collect();
            let mut out_amts: Vec<u64> = batch_outputs.iter().map(|(_, v)| *v).collect();
            out_amts.push(*amount);
            if tentative_change >= CHANGE_DUST_THRESHOLD {
                out_amts.push(tentative_change);
            }
            let mass = estimate_tx_mass(&input_amts, &out_amts);

            if mass <= MAX_SELECTABLE_MASS {
                batch_outputs.push((addr.clone(), *amount));
                batch_inputs_end  = temp_end;
                batch_input_total = temp_total;
            } else {
                break; // this output would push mass over limit — start a new batch
            }
        }

        if batch_outputs.is_empty() {
            break; // no outputs could be processed (UTXO pool exhausted)
        }

        out_cursor  += batch_outputs.len();
        utxo_cursor  = batch_inputs_end;

        let batch_out_sum: u64 = batch_outputs.iter().map(|(_, v)| v).sum();
        let batch_fees = cfg.fee_per_output * (batch_outputs.len() as u64 + 1);
        let batch_change = batch_input_total.saturating_sub(batch_out_sum + batch_fees);

        // ── Build outputs ────────────────────────────────────────────────────
        let mut tx_outputs: Vec<TransactionOutput> = batch_outputs
            .iter()
            .map(|(addr, amount)| TransactionOutput {
                value:             *amount,
                script_public_key: pay_to_address_script(addr),
            })
            .collect();
        if batch_change >= CHANGE_DUST_THRESHOLD {
            tx_outputs.push(TransactionOutput {
                value:             batch_change,
                script_public_key: pay_to_address_script(pool_address),
            });
            info!("Batch {batch_num} change: {batch_change} sompi returned to pool");
        } else if batch_change > 0 {
            warn!("Batch {batch_num} change {batch_change} sompi below dust threshold — consumed as TX fee");
        }

        // ── Build inputs ─────────────────────────────────────────────────────
        let batch_utxos = &sorted_utxos[batch_start_utxo..batch_inputs_end];
        let tx_inputs: Vec<TransactionInput> = batch_utxos
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

        // ── Sign (P2PK Schnorr) ──────────────────────────────────────────────
        let utxo_for_signing: Vec<UtxoEntry> = batch_utxos
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
        let n_inputs          = mutable_tx.tx.inputs.len();

        for i in 0..n_inputs {
            let sig_hash = calc_schnorr_signature_hash(
                &mutable_tx.as_verifiable(), i, SIG_HASH_ALL, &mut reused_values,
            );
            let msg = Message::from_digest_slice(sig_hash.as_bytes().as_slice())
                .context("secp256k1 message from sig_hash")?;
            let sig = keypair.sign_schnorr(msg);
            let mut sig_script = Vec::with_capacity(66);
            sig_script.push(0x41u8); // OP_DATA_65
            sig_script.extend_from_slice(sig.as_ref());
            sig_script.push(SIG_HASH_ALL.to_u8());
            mutable_tx.tx.inputs[i].signature_script = sig_script;
        }

        // ── Submit ───────────────────────────────────────────────────────────
        let rpc_tx = consensus_tx_to_rpc(mutable_tx.tx);
        let tx_id  = rpc
            .submit_transaction(rpc_tx, false)
            .await
            .map_err(|e| anyhow::Error::new(RetryablePayoutError(
                format!("submit_transaction batch {batch_num} RPC failed: {e} — will retry")
            )))?;

        // Record consumed outpoints so subsequent payouts in this cycle skip them.
        for e in batch_utxos {
            spent.insert((e.outpoint.transaction_id, e.outpoint.index));
        }

        info!(
            "Payout batch {batch_num}/{} submitted: {tx_id}  outputs={}  total_out={} sompi",
            (outputs_plan.len() + batch_outputs.len() - 1) / batch_outputs.len().max(1),
            batch_outputs.len(),
            batch_out_sum,
        );

        if first_tx_id.is_none() {
            first_tx_id = Some(tx_id);
        }
    }

    first_tx_id.ok_or_else(|| anyhow::Error::new(RetryablePayoutError(
        "No UTXOs available to cover any payout output — will retry next cycle".into()
    )))
}

// ── UTXO consolidation sweep ──────────────────────────────────────────────────

/// UTXO count above which a consolidation sweep is triggered.
const SWEEP_THRESHOLD: usize = 100;
/// Max inputs consumed per sweep transaction (same limit as payouts).
const SWEEP_BATCH: usize = 84;
/// Approximate mass per P2PK Schnorr input (sompi at 1 sompi/gram min relay fee).
const MASS_PER_INPUT_SOMPI: u64 = 1_200;

/// Consolidate many small UTXOs into one large UTXO sent back to the pool address.
///
/// Returns `Ok(Some(tx_id))` if a sweep was submitted, `Ok(None)` if no sweep
/// was needed (UTXO count ≤ `SWEEP_THRESHOLD`), or an error on RPC failure.
pub async fn consolidate_utxos(
    rpc:          &Arc<GrpcClient>,
    pool_address: &RpcAddress,
    keypair:      &Keypair,
) -> Result<Option<RpcTransactionId>> {
    let current_daa = rpc
        .get_block_dag_info()
        .await
        .map_err(|e| anyhow::anyhow!("consolidate_utxos get_block_dag_info failed: {e}"))?
        .virtual_daa_score;

    let utxo_entries: Vec<_> = rpc
        .get_utxos_by_addresses(vec![pool_address.clone()])
        .await
        .map_err(|e| anyhow::anyhow!("consolidate_utxos get_utxos RPC failed: {e}"))?
        .into_iter()
        .filter(|e| {
            !e.utxo_entry.is_coinbase
                || current_daa.saturating_sub(e.utxo_entry.block_daa_score) >= COINBASE_MATURITY
        })
        .collect();

    if utxo_entries.len() <= SWEEP_THRESHOLD {
        return Ok(None);
    }

    // Pick the SMALLEST UTXOs — these are the ones bloating the UTXO set.
    let mut sorted: Vec<_> = utxo_entries.iter().collect();
    sorted.sort_unstable_by_key(|e| e.utxo_entry.amount);
    let batch: Vec<_> = sorted.into_iter().take(SWEEP_BATCH).collect();

    let total_in: u64 = batch.iter().map(|e| e.utxo_entry.amount).sum();
    let fee:      u64 = MASS_PER_INPUT_SOMPI * batch.len() as u64;
    if total_in <= fee {
        return Ok(None); // UTXOs too dust-like to sweep profitably
    }

    let consolidate_amount = total_in - fee;

    // Build inputs
    let tx_inputs: Vec<TransactionInput> = batch
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

    // Single output back to pool address
    let tx_outputs = vec![TransactionOutput {
        value:             consolidate_amount,
        script_public_key: pay_to_address_script(pool_address),
    }];

    let tx = Transaction::new(0, tx_inputs, tx_outputs, 0, SUBNETWORK_ID_NATIVE, 0, vec![]);

    let utxo_for_signing: Vec<UtxoEntry> = batch
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
    let n_inputs          = mutable_tx.tx.inputs.len();

    for i in 0..n_inputs {
        let sig_hash = calc_schnorr_signature_hash(
            &mutable_tx.as_verifiable(), i, SIG_HASH_ALL, &mut reused_values,
        );
        let msg = Message::from_digest_slice(sig_hash.as_bytes().as_slice())
            .context("secp256k1 message from sig_hash (sweep)")?;
        let sig = keypair.sign_schnorr(msg);

        let mut sig_script = Vec::with_capacity(66);
        sig_script.push(0x41u8);
        sig_script.extend_from_slice(sig.as_ref());
        sig_script.push(SIG_HASH_ALL.to_u8());
        mutable_tx.tx.inputs[i].signature_script = sig_script;
    }

    let rpc_tx = consensus_tx_to_rpc(mutable_tx.tx);
    let tx_id  = rpc
        .submit_transaction(rpc_tx, false)
        .await
        .map_err(|e| anyhow::anyhow!("consolidate_utxos submit_transaction failed: {e}"))?;

    info!(
        "UTXO sweep submitted: {tx_id}  inputs={}  consolidated={} sompi  fee={} sompi",
        n_inputs, consolidate_amount, fee
    );
    Ok(Some(tx_id))
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
