//! Test node runners: a mock WebSocket seed node and an optional Anvil devnet.

use anyhow::{Context, Result};
use borsh::{to_vec, BorshDeserialize};
use futures::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use xenom_miner::rpc::messages::{
    ModelCheckpoint, ModelCheckpointInfo, ModelCheckpointInfoV2, ModelCheckpointV2, RpcEnvelope, RpcRequest, RpcResponse,
    TrainingBatch,
};

/// Anvil availability guard. Tests that need a real EVM skip gracefully when
/// `anvil` is not installed.
pub fn anvil_available() -> bool {
    Command::new("anvil").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

/// Spawn a local Anvil node. Fails if `anvil` is not on PATH.
pub fn spawn_anvil() -> Result<ethers::utils::AnvilInstance> {
    if !anvil_available() {
        anyhow::bail!("anvil binary not found in PATH; install Foundry to run EVM tests");
    }

    let anvil = ethers::utils::Anvil::new().timeout(60_000u64).spawn();

    info!("Anvil spawned at {}", anvil.endpoint());
    Ok(anvil)
}

#[derive(Default, Clone)]
struct MockState {
    submitted_blocks: u64,
    balances: HashMap<String, u64>,
    default_balance: u64,
    reward_per_block: u64,
    close_after: Option<usize>,
}

impl MockState {
    fn balance(&self, address: &str) -> u64 {
        *self.balances.get(address).unwrap_or(&self.default_balance)
    }

    fn record_submission(&mut self, address: &str) {
        self.submitted_blocks += 1;
        // Credit the miner immediately on the mock ledger.
        let current = self.balance(address);
        self.balances.insert(address.to_string(), current + self.reward_per_block);
    }

    fn consume_close(&mut self) -> bool {
        match self.close_after {
            Some(1) => {
                self.close_after = None;
                true
            }
            Some(n) => {
                self.close_after = Some(n - 1);
                false
            }
            None => false,
        }
    }
}

/// A mock WebSocket seed node that implements the Xenom RPC protocol used by
/// `xenom-miner`. It can accept training batches, validate blocks, and keep a
/// mock balance ledger.
pub struct MockSeedNode {
    pub url: String,
    _handle: JoinHandle<()>,
    shutdown: watch::Sender<bool>,
    state: Arc<Mutex<MockState>>,
}

impl MockSeedNode {
    /// Return the balance the node would report for `address`.
    pub fn balance(&self, address: &str) -> u64 {
        self.state.lock().unwrap().balance(address)
    }

    /// Return the number of accepted blocks seen by the node.
    pub fn accepted_blocks(&self) -> u64 {
        self.state.lock().unwrap().submitted_blocks
    }

    /// Close the WebSocket connection after `n` responses on the next connection.
    pub fn close_after_n(&self, n: usize) {
        self.state.lock().unwrap().close_after = Some(n);
    }
}

impl Drop for MockSeedNode {
    fn drop(&mut self) {
        let _ = self.shutdown.send(true);
    }
}

/// Spawn a mock seed node on a free port and return its WebSocket URL.
pub async fn spawn_mock_seed_node() -> Result<MockSeedNode> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let url = format!("ws://127.0.0.1:{}/", port);

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let state = Arc::new(Mutex::new(MockState { default_balance: 10_000, reward_per_block: 10_000_000_000, ..Default::default() }));

    let state_c = Arc::clone(&state);
    let handle = tokio::spawn(async move {
        let mut shutdown = shutdown_rx;
        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        info!("Mock seed node shutting down");
                        break;
                    }
                }
                result = listener.accept() => {
                    match result {
                        Ok((stream, _)) => {
                            let state = Arc::clone(&state_c);
                            tokio::spawn(async move {
                                if let Err(e) = handle_connection(stream, state).await {
                                    warn!("Mock connection closed: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            warn!("Mock seed node accept error: {}", e);
                            break;
                        }
                    }
                }
            }
        }
    });

    Ok(MockSeedNode { url, _handle: handle, shutdown: shutdown_tx, state })
}

async fn handle_connection(stream: tokio::net::TcpStream, state: Arc<Mutex<MockState>>) -> Result<()> {
    let mut ws = accept_async(stream).await.context("Failed to accept WebSocket")?;

    loop {
        match ws.next().await {
            Some(Ok(Message::Binary(bytes))) => {
                let envelope: RpcEnvelope = match RpcEnvelope::try_from_slice(&bytes) {
                    Ok(e) => e,
                    Err(err) => {
                        let response = RpcResponse::Error(format!("borsh decode: {}", err));
                        ws.send(Message::Binary(to_vec(&response)?)).await?;
                        continue;
                    }
                };

                let response = match envelope.payload {
                    RpcRequest::Heartbeat => RpcResponse::Pong,
                    RpcRequest::GetDifficulty => RpcResponse::Difficulty([0u8; 32]),
                    RpcRequest::GetBalance { address } => {
                        let balance = state.lock().unwrap().balance(&address);
                        RpcResponse::Balance(balance)
                    }
                    RpcRequest::GetTrainingBatch { model_id } => {
                        let batch = TrainingBatch {
                            batch_id: 1,
                            model_id,
                            base_checkpoint: [0u8; 32],
                            data_indices: vec![0, 1, 2, 3],
                            target_improvement: 0.01,
                            learning_rate: 0.001,
                        };
                        RpcResponse::TrainingBatch(Some(batch))
                    }
                    RpcRequest::GetModelCheckpoint { model_id } => RpcResponse::ModelCheckpoint(ModelCheckpoint {
                        model_id,
                        base_checkpoint: [0u8; 32],
                        config: b"{}".to_vec(),
                        tokenizer: b"[]".to_vec(),
                        weights: vec![0u8; 64],
                        encrypted: false,
                    }),
                    RpcRequest::SubmitGradients(_) => RpcResponse::GradientAck { new_checkpoint: None },
                    RpcRequest::GetModelCheckpointInfo(req) => {
                        RpcResponse::ModelCheckpointInfo(ModelCheckpointInfo { model_id: req.model_id, base_checkpoint: [0u8; 32] })
                    }
                    RpcRequest::GetModelCheckpointInfoV2(req) => RpcResponse::ModelCheckpointInfoV2(ModelCheckpointInfoV2 {
                        model_id: req.model_id,
                        base_checkpoint: [0u8; 32],
                        base_hash: [0u8; 32],
                    }),
                    RpcRequest::GetModelCheckpointV2(req) => RpcResponse::ModelCheckpointV2(ModelCheckpointV2 {
                        model_id: req.model_id,
                        base_checkpoint: [0u8; 32],
                        base_hash: [0u8; 32],
                        config: b"{}".to_vec(),
                        tokenizer: b"[]".to_vec(),
                        weights: vec![0u8; 64],
                        encrypted: false,
                        is_adapter: false,
                    }),
                    RpcRequest::GetGenomeTrainingBatch(_) => {
                        RpcResponse::Error("mock seed node does not serve genome batches".to_string())
                    }
                    RpcRequest::GetCheckpointPeers(_) => RpcResponse::CheckpointPeers(Vec::new()),
                    RpcRequest::GetTrainingArtifact(_) => {
                        RpcResponse::Error("mock node does not serve LoRA training artifacts".to_string())
                    }
                    RpcRequest::AttestedForward(_) => {
                        RpcResponse::Error("mock node does not serve attested forward passes".to_string())
                    }
                    RpcRequest::SubmitLoRAUpdate(_) => RpcResponse::LoRAUpdateAck { new_checkpoint: None },
                    RpcRequest::SubmitBlock(block) => {
                        // The real proof has 1 version byte + 32 commitments + 32 hash.
                        let valid_proof =
                            block.training_proof.zk_proof.len() == 65 && !block.training_proof.zk_proof.iter().all(|&b| b == 0);

                        if !valid_proof {
                            RpcResponse::Error("invalid zk proof".to_string())
                        } else {
                            state.lock().unwrap().record_submission(&block.miner_address);
                            RpcResponse::BlockHash([1u8; 32])
                        }
                    }
                };

                ws.send(Message::Binary(to_vec(&response)?)).await?;

                if state.lock().unwrap().consume_close() {
                    ws.send(Message::Close(None)).await?;
                    break;
                }
            }
            Some(Ok(Message::Close(_)))
            | Some(Ok(Message::Text(_)))
            | Some(Ok(Message::Ping(_)))
            | Some(Ok(Message::Pong(_)))
            | Some(Ok(Message::Frame(_))) => {
                // Ignore non-binary frames and keep listening.
                continue;
            }
            Some(Err(e)) => {
                warn!("WebSocket error: {}", e);
                break;
            }
            None => break,
        }
    }

    Ok(())
}
