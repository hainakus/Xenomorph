//! Async service wrapper for the unified miner WebSocket server.

use std::path::PathBuf;
use std::sync::Arc;

use kaspa_consensus_core::network::NetworkType;
use kaspa_core::task::service::{AsyncService, AsyncServiceError, AsyncServiceFuture};
use kaspa_core::{info, trace, warn};
use kaspa_p2p_flows::flow_context::FlowContext;
use kaspa_rpc_service::service::RpcCoreService;
use kaspa_utils::networking::ContextualNetAddress;
use kaspa_utils::triggers::SingleTrigger;

use super::coordinator::Coordinator;
use super::websocket_server;

const MINER_WEBSOCKET_SERVICE: &str = "miner-websocket-service";

pub struct MinerWebsocketService {
    listen_address: ContextualNetAddress,
    network_type: NetworkType,
    active_model_id: String,
    models_dir: PathBuf,
    genome_cache_dir: PathBuf,
    genome_file: Option<PathBuf>,
    genome_source_url: String,
    rpc_core_service: Arc<RpcCoreService>,
    genome_fragment_size_bytes: u32,
    genome_pow_activation_daa_score: u64,
    flow_context: Option<Arc<FlowContext>>,
    shutdown: SingleTrigger,
}

impl MinerWebsocketService {
    pub fn new(
        listen_address: ContextualNetAddress,
        network_type: NetworkType,
        active_model_id: String,
        models_dir: PathBuf,
        genome_cache_dir: PathBuf,
        genome_file: Option<PathBuf>,
        genome_source_url: String,
        rpc_core_service: Arc<RpcCoreService>,
        genome_fragment_size_bytes: u32,
        genome_pow_activation_daa_score: u64,
        flow_context: Option<Arc<FlowContext>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            listen_address,
            network_type,
            active_model_id,
            models_dir,
            genome_cache_dir,
            genome_file,
            genome_source_url,
            rpc_core_service,
            genome_fragment_size_bytes,
            genome_pow_activation_daa_score,
            flow_context,
            shutdown: SingleTrigger::new(),
        })
    }
}

impl AsyncService for MinerWebsocketService {
    fn ident(self: Arc<Self>) -> &'static str {
        MINER_WEBSOCKET_SERVICE
    }

    fn start(self: Arc<Self>) -> AsyncServiceFuture {
        Box::pin(async move {
            info!("{} starting", MINER_WEBSOCKET_SERVICE);

            let coordinator = match Coordinator::new(
                self.network_type,
                self.active_model_id.clone(),
                self.models_dir.clone(),
                self.genome_cache_dir.clone(),
                self.genome_file.clone(),
                self.genome_source_url.clone(),
                self.rpc_core_service.clone(),
                self.genome_fragment_size_bytes,
                self.genome_pow_activation_daa_score,
            )
            .await
            {
                Ok(c) => c,
                Err(e) => {
                    return Err(AsyncServiceError::Service(format!("Failed to initialize training coordinator: {}", e)));
                }
            };

            let listen = self.listen_address.to_string();
            let flow_context = self.flow_context.clone();
            let mut server_task =
                tokio::spawn(async move { websocket_server::run_miner_server(&listen, coordinator, flow_context).await });
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
