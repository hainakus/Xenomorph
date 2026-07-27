//! Background sync of active model checkpoints learned from P2P gossip.
//!
//! A unified `xenom` node announces its active checkpoint hash over the Kaspa P2P
//! `CheckpointAnnouncement` message.  This service listens to those announcements
//! and, when it sees a peer with a different (newer) checkpoint for the active
//! model, downloads the checkpoint files directly from that peer's miner WebSocket
//! and loads them into the local `Coordinator`.  This keeps multiple full nodes in
//! sync even when miners connect to different seed/entry points.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use borsh::{to_vec, BorshDeserialize};
use futures_util::{SinkExt, StreamExt};
use kaspa_core::task::service::{AsyncService, AsyncServiceFuture};
use kaspa_core::{info, trace, warn};
use kaspa_p2p_flows::flow_context::FlowContext;
use model_crypto::gossip::{Announcement, GossipRegistry};
use parking_lot::Mutex;
use seed_node::model::model_files::RawModelFiles;
use seed_node::rpc::messages::{
    GetModelCheckpointV2, ModelCheckpointV2 as RpcModelCheckpointV2, RpcRequest, RpcResponse,
};
use tokio::time::{interval, timeout};
use tokio_tungstenite::{connect_async_with_config, tungstenite::protocol::WebSocketConfig, tungstenite::Message};

use super::coordinator::Coordinator;

const SYNC_INTERVAL: Duration = Duration::from_secs(5);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);
const WS_MAX_MESSAGE_SIZE: usize = 1024 * 1024 * 1024;
const CHECKPOINT_SYNC_SERVICE: &str = "checkpoint-sync-service";

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

/// Async service that synchronises the active model checkpoint with peers
/// discovered through P2P gossip.
pub struct CheckpointSyncService {
    coordinator: Coordinator,
    gossip_registry: Arc<Mutex<GossipRegistry>>,
    local_addr: Option<SocketAddr>,
    shutdown: kaspa_utils::triggers::SingleTrigger,
}

impl CheckpointSyncService {
    pub fn new_with_local_addr(
        coordinator: Coordinator,
        flow_context: Arc<FlowContext>,
        local_addr: Option<SocketAddr>,
    ) -> Arc<Self> {
        Arc::new(Self {
            coordinator,
            gossip_registry: flow_context.gossip_registry.clone(),
            local_addr,
            shutdown: kaspa_utils::triggers::SingleTrigger::new(),
        })
    }

    async fn sync_loop(self: Arc<Self>) -> Result<(), kaspa_core::task::service::AsyncServiceError> {
        let mut tick = interval(SYNC_INTERVAL);
        loop {
            tick.tick().await;
            if let Err(e) = self.sync_once().await {
                warn!("Checkpoint sync failed: {}", e);
            }
        }
    }

    async fn sync_once(&self) -> Result<()> {
        let model_id = self.coordinator.active_model_id().to_string();
        let local_hash = self.coordinator.active_weights_hash().await?;

        let announcements = {
            let registry = self.gossip_registry.lock();
            registry.get_by_model(&model_id)
        };

        // Pick the most recent announcement that advertises a different hash and
        // a reachable listen address.  Ignore announcements that point back to us.
        let mut candidate: Option<Announcement> = None;
        for ann in announcements {
            if ann.weights_hash == local_hash.as_bytes() {
                continue;
            }
            if ann.listen_addr.is_none() {
                continue;
            }
            if self.local_addr.is_some() && ann.listen_addr == self.local_addr {
                continue;
            }
            if candidate.as_ref().map_or(true, |c| ann.timestamp > c.timestamp) {
                candidate = Some(ann);
            }
        }

        let Some(candidate) = candidate else {
            trace!("No newer checkpoint announcements to sync");
            return Ok(());
        };

        let peer_addr = candidate.listen_addr.unwrap();
        info!("Syncing checkpoint {} for {} from peer {}", hex::encode(candidate.weights_hash), model_id, peer_addr);

        let checkpoint = fetch_checkpoint_v2(peer_addr, &model_id, None).await?;
        let hash = self.coordinator.load_external_checkpoint(model_id, raw_model_files_from_v2(checkpoint)).await?;
        info!(
            "Loaded synced checkpoint {} for {}",
            hex::encode(hash.as_bytes()),
            self.coordinator.active_model_id()
        );
        Ok(())
    }
}

impl AsyncService for CheckpointSyncService {
    fn ident(self: Arc<Self>) -> &'static str {
        CHECKPOINT_SYNC_SERVICE
    }

    fn start(self: Arc<Self>) -> AsyncServiceFuture {
        Box::pin(async move {
            let shutdown_signal = self.shutdown.listener.clone();
            tokio::select! {
                _ = shutdown_signal => Ok(()),
                res = self.sync_loop() => {
                    warn!("{} sync loop exited unexpectedly", CHECKPOINT_SYNC_SERVICE);
                    res
                }
            }
        })
    }

    fn signal_exit(self: Arc<Self>) {
        self.shutdown.trigger.trigger();
    }

    fn stop(self: Arc<Self>) -> AsyncServiceFuture {
        Box::pin(async move { Ok(()) })
    }
}

/// Connect to `peer` over WebSocket, request `GetModelCheckpointV2`, and return the files.
async fn fetch_checkpoint_v2(peer: SocketAddr, model_id: &str, cached_base_hash: Option<[u8; 32]>) -> Result<RpcModelCheckpointV2> {
    let url = format!("ws://{}", peer);
    let (mut ws, _) = connect_async_with_config(&url, Some(ws_config()), false)
        .await
        .with_context(|| format!("Failed to connect to peer WebSocket {} for checkpoint sync", peer))?;

    let request_id = 1u64;
    let request = seed_node::rpc::messages::RpcEnvelope {
        request_id,
        payload: RpcRequest::GetModelCheckpointV2(GetModelCheckpointV2 {
            model_id: model_id.to_string(),
            cached_base_hash,
        }),
    };
    let req_bytes = to_vec(&request).context("Failed to serialize checkpoint request")?;
    ws.send(Message::Binary(req_bytes)).await.with_context(|| format!("Failed to send checkpoint request to {}", peer))?;

    let response = timeout(DOWNLOAD_TIMEOUT, ws.next())
        .await
        .with_context(|| format!("Timeout waiting for checkpoint from {}", peer))?
        .with_context(|| format!("WebSocket stream closed by peer {} before checkpoint response", peer))?
        .with_context(|| format!("WebSocket error from peer {}", peer))?;

    match response {
        Message::Binary(bytes) => match RpcResponse::try_from_slice(&bytes)
            .with_context(|| format!("Failed to deserialize checkpoint response from {}", peer))?
        {
            RpcResponse::ModelCheckpointV2(cp) => Ok(cp),
            RpcResponse::Error(msg) => bail!("Peer {} returned error: {}", peer, msg),
            other => bail!("Unexpected response from peer {}: {:?}", peer, other),
        },
        Message::Close(_) => bail!("Peer {} closed connection before returning checkpoint", peer),
        _ => bail!("Peer {} returned non-binary WebSocket message", peer),
    }
}

fn raw_model_files_from_v2(cp: RpcModelCheckpointV2) -> RawModelFiles {
    RawModelFiles { config: cp.config, tokenizer: cp.tokenizer, weights: cp.weights }
}
