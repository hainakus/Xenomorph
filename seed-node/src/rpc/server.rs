use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use borsh::{to_vec, BorshDeserialize};
use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::time::interval;
use tokio_tungstenite::accept_async_with_config;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::genome::{GenomeBatchGenerator, GenomeStorage};
use crate::model::manager::ModelManager;
use crate::p2p::P2pGossipHandle;
use crate::rpc::client::XenomorphRpcClient;
use crate::rpc::messages::{
    GenomeTrainingBatchMsg, GetCheckpointPeers, GetGenomeTrainingBatch, GetModelCheckpointInfo, GetModelCheckpointInfoV2,
    ModelCheckpointInfoV2, ModelCheckpointV2, PeerAnnouncement, RpcEnvelope, RpcRequest, RpcResponse, TrainingBatch,
};

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
    p2p_gossip: Option<Arc<P2pGossipHandle>>,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await.map_err(|e| anyhow!("Failed to bind miner server {}: {}", addr, e))?;
    let bound: SocketAddr = listener.local_addr()?;
    info!("Miner WebSocket server listening on {}", bound);

    // Per-request nonce so every genome batch is drawn from a different RNG state.
    let batch_counter = Arc::new(AtomicU64::new(1));

    while let Ok((stream, peer)) = listener.accept().await {
        let mm = model_manager.clone();
        let gs = genome_storage.clone();
        let xc = xenomorph_client.clone();
        let pg = p2p_gossip.clone();
        let bc = batch_counter.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, mm, gs, xc, pg, bc).await {
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
    p2p_gossip: Option<Arc<P2pGossipHandle>>,
    batch_counter: Arc<AtomicU64>,
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

                // While a long-running request (e.g. SubmitGradients) is being
                // processed, send WebSocket ping frames every 15 seconds. This
                // keeps NATs/middleboxes from closing the connection and lets the
                // client extend its wait timeout.
                let mut ping_interval = interval(Duration::from_secs(15));
                ping_interval.tick().await; // skip the immediate first tick

                let mut request_fut = Box::pin(handle_request(
                    envelope.payload,
                    model_manager.clone(),
                    genome_storage.clone(),
                    xenomorph_client.clone(),
                    p2p_gossip.clone(),
                    batch_counter.clone(),
                ));

                let response = loop {
                    tokio::select! {
                        _ = ping_interval.tick() => {
                            if let Err(e) = ws.send(Message::Ping(vec![])).await {
                                warn!("Failed to send ping to miner: {}", e);
                                return Err(e.into());
                            }
                        }
                        resp = request_fut.as_mut() => break resp,
                    }
                };

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
    p2p_gossip: Option<Arc<P2pGossipHandle>>,
    batch_counter: Arc<AtomicU64>,
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
        RpcRequest::GetGenomeTrainingBatch(request) => {
            handle_genome_batch_request(request, genome_storage, model_manager, batch_counter).await
        }
        RpcRequest::GetModelCheckpointInfo(GetModelCheckpointInfo { model_id }) => {
            RpcResponse::ModelCheckpointInfo(super::messages::ModelCheckpointInfo {
                model_id: model_id.clone(),
                base_checkpoint: get_checkpoint(&model_manager, &model_id).await,
            })
        }
        RpcRequest::GetModelCheckpoint { model_id } => match model_manager.get_encrypted_model_checkpoint(&model_id).await {
            Ok((checkpoint, files)) => RpcResponse::ModelCheckpoint(super::messages::ModelCheckpoint {
                model_id,
                base_checkpoint: checkpoint.weights_hash,
                config: files.config,
                tokenizer: files.tokenizer,
                weights: files.weights,
                encrypted: true,
            }),
            Err(e) => RpcResponse::Error(format!("Failed to get model checkpoint: {}", e)),
        },
        RpcRequest::GetModelCheckpointInfoV2(GetModelCheckpointInfoV2 { model_id }) => {
            match model_manager.get_model_checkpoint_info_v2(&model_id).await {
                Ok((combined, base_hash)) => {
                    RpcResponse::ModelCheckpointInfoV2(ModelCheckpointInfoV2 { model_id, base_checkpoint: combined, base_hash })
                }
                Err(e) => RpcResponse::Error(format!("Failed to get model checkpoint info: {}", e)),
            }
        }
        RpcRequest::GetModelCheckpointV2(request) => {
            match model_manager.get_encrypted_model_checkpoint_v2(&request.model_id, request.cached_base_hash).await {
                Ok((files, combined, base_hash, is_adapter)) => RpcResponse::ModelCheckpointV2(ModelCheckpointV2 {
                    model_id: request.model_id,
                    base_checkpoint: combined,
                    base_hash,
                    config: files.config,
                    tokenizer: files.tokenizer,
                    weights: files.weights,
                    encrypted: true,
                    is_adapter,
                }),
                Err(e) => RpcResponse::Error(format!("Failed to get model checkpoint: {}", e)),
            }
        }
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
        RpcRequest::SubmitGradients(update) => match model_manager.submit_gradients(&update).await {
            Ok(new_checkpoint) => RpcResponse::GradientAck { new_checkpoint },
            Err(e) => {
                warn!("Failed to submit gradients for {}: {}", update.model_id, e);
                RpcResponse::Error(format!("Failed to submit gradients: {}", e))
            }
        },
        RpcRequest::GetCheckpointPeers(GetCheckpointPeers { weights_hash, .. }) => match p2p_gossip {
            Some(gossip) => {
                let announcements = gossip.known_peers(&weights_hash).await;
                let peers = announcements.iter().map(peer_announcement_from).collect();
                RpcResponse::CheckpointPeers(peers)
            }
            None => RpcResponse::Error("P2P gossip not enabled on this seed-node".to_string()),
        },
        RpcRequest::GetBalance { .. } => RpcResponse::Balance(10_000),
        RpcRequest::GetDifficulty => RpcResponse::Difficulty([0u8; 32]),
        RpcRequest::Heartbeat => RpcResponse::Pong,
    }
}

fn peer_announcement_from(ann: &model_crypto::gossip::Announcement) -> PeerAnnouncement {
    PeerAnnouncement {
        model_id: ann.model_id.clone(),
        weights_hash: ann.weights_hash,
        cid: ann.cid,
        timestamp: ann.timestamp,
        is_genome: ann.is_genome,
        node_address: ann.node_address.clone(),
        public_key: ann.public_key,
        listen_addr: ann.listen_addr.map(|addr| addr.to_string()),
        signature: ann.signature,
    }
}

fn validate_miner_address(address: &str) -> Result<()> {
    let _ = kaspa_addresses::Address::try_from(address).map_err(|e| anyhow!("{}", e))?;
    Ok(())
}

async fn handle_genome_batch_request(
    request: GetGenomeTrainingBatch,
    genome_storage: Arc<RwLock<GenomeStorage>>,
    model_manager: Arc<ModelManager>,
    batch_counter: Arc<AtomicU64>,
) -> RpcResponse {
    // The seed-node auto-discovers the genome archive in its local cache or falls
    // back to the canonical GitHub Releases URL (overridable via XENO_GENOME_URL).
    let source = String::new();

    let archive = match genome_storage.write().await.get_or_load(request.genome_merkle_root, &source).await {
        Ok(archive) => archive,
        Err(e) => {
            return RpcResponse::Error(format!("Failed to load genome archive: {}", e));
        }
    };

    // Derive the RNG seed from the genome merkle root plus a per-request nonce.
    // The merkle root identifies which .xenom archive to use; the nonce ensures
    // each batch extracts a different region of the real DNA.
    let batch_nonce = batch_counter.fetch_add(1, Ordering::Relaxed);
    let mut seed_input = Vec::with_capacity(40);
    seed_input.extend_from_slice(&request.genome_merkle_root);
    seed_input.extend_from_slice(&batch_nonce.to_le_bytes());
    let seed = *blake3::hash(&seed_input).as_bytes();

    let mut generator = GenomeBatchGenerator::new(archive, seed);
    let seq_len_bases = 512usize.saturating_mul(4);
    let mut batch = generator.generate_batch(request.preferred_batch_size.max(1), seq_len_bases);
    batch.batch_id = batch_nonce;
    batch.model_id = request.model_id.clone();

    let sequences = generator.extract_sequences(&batch);
    let base_checkpoint = get_checkpoint(&model_manager, &request.model_id).await;

    RpcResponse::GenomeTrainingBatch(GenomeTrainingBatchMsg { batch, sequences, base_checkpoint })
}

async fn get_checkpoint(model_manager: &ModelManager, model_id: &str) -> [u8; 32] {
    match model_manager.load_model(model_id).await {
        Ok(info) => info.checkpoint.weights_hash,
        Err(_) => [0u8; 32],
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use model_crypto::gossip::{Announcement, GossipRegistry};

    use super::*;

    #[test]
    fn test_peer_announcement_from_registry() {
        let mut registry = GossipRegistry::new();
        let hash = [1u8; 32];
        let ann = Announcement {
            model_id: "multimolecule/dnabert2".to_string(),
            weights_hash: hash,
            cid: [2u8; 32],
            timestamp: 1234,
            is_genome: false,
            node_address: "xnom:test".to_string(),
            public_key: [3u8; 33],
            listen_addr: Some(SocketAddr::from(([127, 0, 0, 1], 8443))),
            signature: [4u8; 64],
        };
        registry.insert(ann.clone());

        let found = registry.get(&hash);
        assert_eq!(found.len(), 1);

        let peer = peer_announcement_from(&found[0]);
        assert_eq!(peer.model_id, ann.model_id);
        assert_eq!(peer.weights_hash, hash);
        assert_eq!(peer.listen_addr, Some("127.0.0.1:8443".to_string()));
    }
}
