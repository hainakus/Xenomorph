//! WebSocket server for miner connections.
//!
//! This is the unified node's replacement for the standalone seed-node WebSocket
//! server. It speaks the same Borsh-over-WebSocket protocol as `xenom-miner`.

use std::net::SocketAddr;

use anyhow::{anyhow, Result};
use borsh::{to_vec, BorshDeserialize};
use futures_util::{SinkExt, StreamExt};
use kaspa_core::{info, warn};
use kaspa_p2p_flows::flow_context::FlowContext;
use seed_node::rpc::messages::{GetModelCheckpointInfo, GetModelCheckpointInfoV2, RpcEnvelope, RpcRequest, RpcResponse};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::accept_async_with_config;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;

use super::coordinator::Coordinator;

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

/// Run the miner WebSocket server on `addr` using `coordinator` for state.
pub async fn run_miner_server(addr: &str, coordinator: Coordinator, flow_context: Option<Arc<FlowContext>>) -> Result<()> {
    let listener = TcpListener::bind(addr).await.map_err(|e| anyhow!("Failed to bind miner server {}: {}", addr, e))?;
    let bound: SocketAddr = listener.local_addr()?;
    info!("Unified miner WebSocket server listening on {}", bound);

    while let Ok((stream, peer)) = listener.accept().await {
        let coord = coordinator.clone();
        let ctx = flow_context.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, peer, coord, ctx).await {
                warn!("Miner WebSocket connection from {} closed: {}", peer, e);
            }
        });
    }

    Ok(())
}

async fn handle_connection(
    stream: TcpStream,
    peer: SocketAddr,
    coordinator: Coordinator,
    flow_context: Option<Arc<FlowContext>>,
) -> Result<()> {
    let mut ws = accept_async_with_config(stream, Some(ws_config())).await?;

    while let Some(msg) = ws.next().await {
        let msg = msg?;
        match msg {
            Message::Binary(bytes) => {
                let envelope: RpcEnvelope = match BorshDeserialize::try_from_slice(&bytes) {
                    Ok(env) => env,
                    Err(e) => {
                        warn!("Failed to deserialize miner request from {}: {}", peer, e);
                        continue;
                    }
                };

                let response = handle_request(envelope.payload, &coordinator, flow_context.clone()).await;
                let resp_bytes = match to_vec(&response) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        warn!("Failed to serialize miner response for {}: {}", peer, e);
                        continue;
                    }
                };

                if let Err(e) = ws.send(Message::Binary(resp_bytes)).await {
                    warn!("Failed to send miner response to {}: {}", peer, e);
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    Ok(())
}

async fn handle_request(req: RpcRequest, coordinator: &Coordinator, flow_context: Option<Arc<FlowContext>>) -> RpcResponse {
    match req {
        RpcRequest::GetTrainingBatch { model_id } => coordinator.get_training_batch(model_id).await,
        RpcRequest::GetGenomeTrainingBatch(request) => coordinator.get_genome_training_batch(request).await,
        RpcRequest::GetModelCheckpoint { model_id } => coordinator.get_model_checkpoint(model_id).await,
        RpcRequest::GetModelCheckpointInfo(GetModelCheckpointInfo { model_id }) => {
            coordinator.get_model_checkpoint_info(model_id).await
        }
        RpcRequest::GetModelCheckpointInfoV2(GetModelCheckpointInfoV2 { model_id }) => {
            coordinator.get_model_checkpoint_info_v2(model_id).await
        }
        RpcRequest::GetModelCheckpointV2(request) => {
            coordinator.get_model_checkpoint_v2(request.model_id, request.cached_base_hash).await
        }
        RpcRequest::SubmitBlock(block) => coordinator.submit_block(block).await,
        RpcRequest::SubmitGradients(update) => match coordinator.submit_gradients(&update).await {
            Ok(Some(new_checkpoint)) => {
                if let Some(ctx) = flow_context {
                    // Announce the new active checkpoint over P2P gossip so other nodes
                    // (e.g. standalone seed-nodes) can discover it.  `cid` is currently a
                    // placeholder equal to the weights hash until IPFS/HTTP content IDs are wired.
                    ctx.announce_checkpoint(update.model_id.clone(), new_checkpoint, new_checkpoint, None).await;
                }
                RpcResponse::GradientAck { new_checkpoint: Some(new_checkpoint) }
            }
            Ok(None) => RpcResponse::GradientAck { new_checkpoint: None },
            Err(e) => {
                warn!("Failed to submit gradients for {}: {}", update.model_id, e);
                RpcResponse::Error(format!("Failed to submit gradients: {}", e))
            }
        },
        RpcRequest::GetBalance { .. } => RpcResponse::Balance(10_000),
        RpcRequest::GetDifficulty => RpcResponse::Difficulty([0u8; 32]),
        RpcRequest::Heartbeat => RpcResponse::Pong,
    }
}
