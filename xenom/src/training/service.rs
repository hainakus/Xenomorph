//! Async service wrapper for the unified miner WebSocket server.

use std::net::SocketAddr;
use std::sync::Arc;

use kaspa_core::task::service::{AsyncService, AsyncServiceError, AsyncServiceFuture};
use kaspa_core::{info, trace, warn};
use kaspa_p2p_flows::flow_context::FlowContext;
use kaspa_utils::networking::ContextualNetAddress;
use kaspa_utils::triggers::SingleTrigger;

use super::coordinator::Coordinator;
use super::websocket_server;

const MINER_WEBSOCKET_SERVICE: &str = "miner-websocket-service";

pub struct MinerWebsocketService {
    listen_address: ContextualNetAddress,
    coordinator: Arc<Coordinator>,
    flow_context: Option<Arc<FlowContext>>,
    /// Optional public listen address to advertise in P2P gossip so peers can
    /// fetch checkpoints directly from this node.
    advertised_address: Option<SocketAddr>,
    shutdown: SingleTrigger,
}

impl MinerWebsocketService {
    pub fn new(
        listen_address: ContextualNetAddress,
        coordinator: Arc<Coordinator>,
        flow_context: Option<Arc<FlowContext>>,
        advertised_address: Option<SocketAddr>,
    ) -> Arc<Self> {
        Arc::new(Self { listen_address, coordinator, flow_context, advertised_address, shutdown: SingleTrigger::new() })
    }
}

impl AsyncService for MinerWebsocketService {
    fn ident(self: Arc<Self>) -> &'static str {
        MINER_WEBSOCKET_SERVICE
    }

    fn start(self: Arc<Self>) -> AsyncServiceFuture {
        Box::pin(async move {
            info!("{} starting", MINER_WEBSOCKET_SERVICE);

            let listen = self.listen_address.to_string();
            let coordinator = self.coordinator.as_ref().clone();
            let flow_context = self.flow_context.clone();
            let advertised_address = self.advertised_address;
            let mut server_task = tokio::spawn(async move {
                websocket_server::run_miner_server(&listen, coordinator, flow_context, advertised_address).await
            });
            let shutdown_signal = self.shutdown.listener.clone();

            tokio::select! {
                _ = shutdown_signal => {
                    server_task.abort();
                    info!("{} shutting down", MINER_WEBSOCKET_SERVICE);
                    Ok(())
                }
                res = &mut server_task => {
                    match res {
                        Ok(Ok(())) => {
                            warn!("{} server task exited unexpectedly", MINER_WEBSOCKET_SERVICE);
                            Ok(())
                        }
                        Ok(Err(e)) => {
                            warn!("{} server loop error: {}", MINER_WEBSOCKET_SERVICE, e);
                            Err(AsyncServiceError::Service(format!("Miner websocket server error: {}", e)))
                        }
                        Err(e) => {
                            warn!("{} server task panicked: {}", MINER_WEBSOCKET_SERVICE, e);
                            Err(AsyncServiceError::Service(format!("Miner websocket server panicked: {}", e)))
                        }
                    }
                }
            }
        })
    }

    fn signal_exit(self: Arc<Self>) {
        self.shutdown.trigger.trigger();
    }

    fn stop(self: Arc<Self>) -> AsyncServiceFuture {
        Box::pin(async move {
            trace!("{} stopped", MINER_WEBSOCKET_SERVICE);
            Ok(())
        })
    }
}
