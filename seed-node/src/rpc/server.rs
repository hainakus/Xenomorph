use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use borsh_miner::{to_vec, BorshDeserialize};
use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio_tungstenite::accept_async_with_config;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::genome::{GenomeBatchGenerator, GenomeStorage};
use crate::model::manager::ModelManager;
use crate::rpc::client::XenomorphRpcClient;
use crate::rpc::messages::{GenomeTrainingBatchMsg, GetGenomeTrainingBatch, RpcEnvelope, RpcRequest, RpcResponse, TrainingBatch};

/// Allow WebSocket messages up to 1 GiB so model checkpoints (config + tokenizer + weights) fit.
const WS_MAX_MESSAGE_SIZE: usize = 1024 * 1024 * 1024;

fn ws_config() -> WebSocketConfig {
    #[allow(deprecated)]
    WebSocketConfig {
        max_send_queue: None,
        write_buffer_size: 128 * 1024,
        max_write_buffer_size: usize::MAX,
        max_message_size: Some(WS_MAX_MESSAGE_SIZE),
        max_frame_size: Some(WS_MAX_MESSAGE_SIZE),
        accept_unmasked_frames: false,
    }
}

/// Start a WebSocket server for miner connections.
pub async fn run_miner_server(
    addr: &str,
    model_manager: Arc<ModelManager>,
    genome_storage: Arc<RwLock<GenomeStorage>>,
    xenomorph_client: Option<Arc<XenomorphRpcClient>>,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await.map_err(|e| anyhow!("Failed to bind miner server {}: {}", addr, e))?;
    let bound: SocketAddr = listener.local_addr()?;
    info!("Miner WebSocket server listening on {}", bound);

    while let Ok((stream, peer)) = listener.accept().await {
        let mm = model_manager.clone();
        let gs = genome_storage.clone();
        let xc = xenomorph_client.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, mm, gs, xc).await {
                warn!("Miner WebSocket connection from {} closed: {}", peer, e);
            }
        });
    }

    Ok(())
}

async fn handle_connection(
    stream: TcpStream,
    model_manager: Arc<ModelManager>,
    genome_storage: Arc<RwLock<GenomeStorage>>,
    xenomorph_client: Option<Arc<XenomorphRpcClient>>,
) -> Result<()> {
    let mut ws = accept_async_with_config(stream, Some(ws_config())).await?;

    while let Some(msg) = ws.next().await {
        let msg = msg?;
        match msg {
            Message::Binary(bytes) => {
                let envelope: RpcEnvelope = match BorshDeserialize::try_from_slice(&bytes) {
                    Ok(env) => env,
                    Err(e) => {
                        warn!("Failed to deserialize miner request: {}", e);
                        continue;
                    }
                };

                let response =
                    handle_request(envelope.payload, model_manager.clone(), genome_storage.clone(), xenomorph_client.clone()).await;
                let resp_bytes = match to_vec(&response) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        warn!("Failed to serialize miner response: {}", e);
                        continue;
                    }
                };

                if let Err(e) = ws.send(Message::Binary(resp_bytes)).await {
                    warn!("Failed to send miner response: {}", e);
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    Ok(())
}

async fn handle_request(
    req: RpcRequest,
    model_manager: Arc<ModelManager>,
    genome_storage: Arc<RwLock<GenomeStorage>>,
    xenomorph_client: Option<Arc<XenomorphRpcClient>>,
) -> RpcResponse {
    match req {
        RpcRequest::GetTrainingBatch { model_id } => {
            let base_checkpoint = get_checkpoint(&model_manager, &model_id).await;
            RpcResponse::TrainingBatch(Some(TrainingBatch {
                batch_id: 1,
                model_id,
                base_checkpoint,
                data_indices: (0..4).collect(),
                target_improvement: 0.01,
                learning_rate: 0.01,
            }))
        }
        RpcRequest::GetGenomeTrainingBatch(request) => handle_genome_batch_request(request, genome_storage).await,
        RpcRequest::GetModelCheckpoint { model_id } => match model_manager.get_model_checkpoint(&model_id).await {
            Ok((checkpoint, files)) => RpcResponse::ModelCheckpoint(super::messages::ModelCheckpoint {
                model_id,
                base_checkpoint: checkpoint.weights_hash,
                config: files.config,
                tokenizer: files.tokenizer,
                weights: files.weights,
            }),
            Err(e) => RpcResponse::Error(format!("Failed to get model checkpoint: {}", e)),
        },
        RpcRequest::SubmitBlock(block) => {
            // Validate the miner's address is a syntactically valid Kaspa/Xenom address
            // before we sign anything or forward it. We cannot verify the signature here
            // because the seed-node does not have the miner's public key.
            if let Err(e) = validate_miner_address(&block.miner_address) {
                warn!("Rejected block {} from invalid miner address {}: {}", block.header.block_number, block.miner_address, e);
                return RpcResponse::Error(format!("Invalid miner address: {}", e));
            }

            // Try to propagate the training proof to the Xenomorph full node. This is a
            // best-effort bridge: the full node must expose the Borsh training-block
            // listener on XENO_NODE_RPC for the chain to actually advance.
            if let Some(client) = xenomorph_client {
                match to_vec(&block.training_proof) {
                    Ok(proof_bytes) => {
                        let block_height = block.header.block_number;
                        let model_id = block.model_id.clone();
                        let miner_address = block.miner_address.clone();
                        match client.submit_training_block(&model_id, &miner_address, proof_bytes, block_height).await {
                            Ok(resp) if resp.accepted => {
                                info!("Full node accepted training block {} (hash {})", block_height, hex::encode(resp.block_hash));
                                return RpcResponse::BlockHash(resp.block_hash);
                            }
                            Ok(resp) => {
                                warn!("Full node rejected training block {} (accepted={})", block_height, resp.accepted);
                                // Fall through to the local hash response so the miner can keep submitting.
                            }
                            Err(e) => {
                                warn!("Failed to forward training block {} to full node: {}", block.header.block_number, e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to serialize training proof for forwarding: {}", e);
                    }
                }
            } else {
                warn!("No Xenomorph node client configured; training block {} will not advance the chain", block.header.block_number);
            }

            // Local acknowledgement: compute the block hash ourselves so the miner can
            // continue its local tip chain. This does NOT mean the full node accepted it.
            let block_bytes = match to_vec(&block) {
                Ok(bytes) => bytes,
                Err(e) => return RpcResponse::Error(format!("Failed to serialize block: {}", e)),
            };
            let hash = blake3::hash(&block_bytes);
            let mut block_hash = [0u8; 32];
            block_hash.copy_from_slice(hash.as_bytes());
            RpcResponse::BlockHash(block_hash)
        }
        RpcRequest::GetBalance { .. } => RpcResponse::Balance(10_000),
        RpcRequest::GetDifficulty => RpcResponse::Difficulty([0u8; 32]),
        RpcRequest::Heartbeat => RpcResponse::Pong,
    }
}

fn validate_miner_address(address: &str) -> Result<()> {
    let _ = kaspa_addresses::Address::try_from(address).map_err(|e| anyhow!("{}", e))?;
    Ok(())
}

async fn handle_genome_batch_request(request: GetGenomeTrainingBatch, genome_storage: Arc<RwLock<GenomeStorage>>) -> RpcResponse {
    // The seed-node auto-discovers the genome archive in its local cache or falls
    // back to the canonical GitHub Releases URL (overridable via XENO_GENOME_URL).
    let source = String::new();

    let archive = match genome_storage.write().await.get_or_load(request.genome_merkle_root, &source).await {
        Ok(archive) => archive,
        Err(e) => {
            return RpcResponse::Error(format!("Failed to load genome archive: {}", e));
        }
    };

    // Deterministic seed derived from the genome merkle root.
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&request.genome_merkle_root);

    let mut generator = GenomeBatchGenerator::new(archive, seed);
    let mut batch = generator.generate_batch(request.preferred_batch_size, 128);
    batch.model_id = request.model_id;

    let sequences = generator.extract_sequences(&batch);

    RpcResponse::GenomeTrainingBatch(GenomeTrainingBatchMsg { batch, sequences })
}

async fn get_checkpoint(model_manager: &ModelManager, model_id: &str) -> [u8; 32] {
    match model_manager.load_model(model_id).await {
        Ok(info) => info.checkpoint.weights_hash,
        Err(_) => [0u8; 32],
    }
}
