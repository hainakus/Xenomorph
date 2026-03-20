use std::sync::Arc;

use clap::{Arg, Command};
use kaspa_grpc_client::GrpcClient;
use kaspa_rpc_core::api::rpc::RpcApi;
use xenom_evm_core::{Address, EvmChain, U256};
use xenom_evm_rpc::start_rpc_server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();

    let chain_id: u64 = m
        .get_one::<String>("chain-id")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1337);

    let rpc_addr = m
        .get_one::<String>("rpc-addr")
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:8545".to_string());

    let state_dir = m.get_one::<String>("state-dir").cloned();
    let chain = Arc::new(if let Some(ref dir) = state_dir {
        EvmChain::new_with_persistence(chain_id, std::path::Path::new(dir))
    } else {
        EvmChain::new(chain_id)
    });

    // Pre-fund well-known devnet address (Hardhat/Anvil account #0)
    // Skip if balance is already non-zero (loaded from snapshot).
    if m.get_flag("devnet") {
        let devnet_addr = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
        let addr_bytes = hex::decode(devnet_addr.trim_start_matches("0x")).unwrap();
        let addr = Address::from_slice(&addr_bytes);
        if chain.balance(addr) == U256::ZERO {
            let amount = U256::from(10_000u64) * U256::from(1_000_000_000_000_000_000u128);
            chain.fund(addr, amount);
            log::info!("devnet: pre-funded {devnet_addr} with 10000 ETH");
        } else {
            log::info!("devnet: {devnet_addr} already funded (snapshot), skipping pre-fund");
        }
    }

    let block_time_ms: u64 = m
        .get_one::<String>("block-time")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    let l1_node = m.get_one::<String>("l1-node").cloned();

    log::info!("Xenom EVM node starting");
    log::info!("  chain_id:   {chain_id}");
    log::info!("  rpc:        http://{rpc_addr}");
    if let Some(ref dir) = state_dir {
        log::info!("  state:      {dir}/evm-state-{chain_id}.rocksdb");
    } else {
        log::info!("  state:      volatile (no --state-dir)");
    }
    if let Some(ref url) = l1_node {
        log::info!("  sequencer:  DAA-tied via {url}");
    } else {
        log::info!("  sequencer:  fixed-timer {}ms", block_time_ms);
    }

    let handle = start_rpc_server(Arc::clone(&chain), &rpc_addr).await?;

    let miner_chain = Arc::clone(&chain);

    if let Some(l1_url) = l1_node {
        // ── DAA-tied sequencer ────────────────────────────────────────────────
        // Mine one EVM block per Xenom L1 block (DAA score increment).
        tokio::spawn(async move {
            let url = if l1_url.starts_with("grpc://") {
                l1_url.clone()
            } else {
                format!("grpc://{l1_url}")
            };

            // Retry connection loop
            loop {
                match GrpcClient::connect(url.clone()).await {
                    Ok(rpc) => {
                        log::info!("DAA sequencer: connected to {url}");
                        let mut last_daa: u64 = 0;
                        loop {
                            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                            match rpc.get_block_dag_info().await {
                                Ok(info) => {
                                    let daa = info.virtual_daa_score;
                                    if daa > last_daa {
                                        let delta = daa - last_daa;
                                        // Mine one block per L1 block, catching up if needed
                                        for _ in 0..delta.min(10) {
                                            let (blk, root) = miner_chain.mine_block();
                                            if miner_chain.pending_count() > 0 || blk % 100 == 0 {
                                                log::info!(
                                                    "⛏  EVM block {blk} (L1 daa={daa}) root={root}"
                                                );
                                            }
                                        }
                                        last_daa = daa;
                                    }
                                }
                                Err(e) => {
                                    log::warn!("DAA sequencer: poll error: {e}");
                                    break; // reconnect
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("DAA sequencer: connect failed: {e} — retrying in 5s");
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                }
            }
        });
    } else {
        // ── Fixed-timer sequencer (default) ──────────────────────────────────
        tokio::spawn(async move {
            let interval = std::time::Duration::from_millis(block_time_ms);
            loop {
                tokio::time::sleep(interval).await;
                let (blk, root) = miner_chain.mine_block();
                if miner_chain.pending_count() > 0 || blk % 50 == 0 {
                    log::debug!("⛏  block {blk} root={root}");
                }
            }
        });
    }

    tokio::signal::ctrl_c().await?;
    log::info!("Shutting down...");
    handle.stop()?;
    Ok(())
}

fn cli() -> Command {
    Command::new("xenom-evm-node")
        .about("Xenom EVM L2 — embedded revm node with Ethereum-compatible JSON-RPC")
        .arg(
            Arg::new("chain-id")
                .long("chain-id")
                .value_name("ID")
                .default_value("1337")
                .help("EVM chain ID (default 1337 for devnet)"),
        )
        .arg(
            Arg::new("rpc-addr")
                .long("rpc-addr")
                .value_name("HOST:PORT")
                .default_value("127.0.0.1:8545")
                .help("JSON-RPC listen address"),
        )
        .arg(
            Arg::new("devnet")
                .long("devnet")
                .action(clap::ArgAction::SetTrue)
                .help("Pre-fund well-known devnet address (Hardhat/Anvil account #0)"),
        )
        .arg(
            Arg::new("block-time")
                .long("block-time")
                .value_name("MS")
                .default_value("2000")
                .help("Fixed-timer block interval in ms (ignored when --l1-node is set)"),
        )
        .arg(
            Arg::new("l1-node")
                .long("l1-node")
                .value_name("HOST:PORT")
                .help("Xenom L1 gRPC endpoint (e.g. localhost:36669). \
                       When set, mines one EVM block per L1 DAA score increment \
                       instead of using the fixed timer."),
        )
        .arg(
            Arg::new("state-dir")
                .long("state-dir")
                .value_name("PATH")
                .help("Directory for persistent state snapshots. \
                       Stores RocksDB state at evm-state-{chain_id}.rocksdb and \
                       restores state on restart. Omit for volatile (dev) mode."),
        )
}
