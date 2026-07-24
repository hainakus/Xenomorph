//! Async service wrapper for the gRPC inference server.
//!
//! Reuses `seed_node::serving::inference::InferenceService` so the unified `xenom` node
//! can expose the same `xenom.inference.Inference` API as the standalone seed-node.

use std::net::SocketAddr;
use std::sync::Arc;

use kaspa_core::task::service::{AsyncService, AsyncServiceError, AsyncServiceFuture};
use kaspa_core::{info, warn};
use kaspa_utils::networking::ContextualNetAddress;
use kaspa_utils::triggers::SingleTrigger;
use tonic::transport::Server;

use super::coordinator::Coordinator;

const INFERENCE_GRPC_SERVICE: &str = "inference-grpc-service";

pub struct InferenceGrpcService {
    listen_address: ContextualNetAddress,
    coordinator: Arc<Coordinator>,
    shutdown: SingleTrigger,
}

impl InferenceGrpcService {
    pub fn new(listen_address: ContextualNetAddress, coordinator: Arc<Coordinator>) -> Arc<Self> {
        Arc::new(Self { listen_address, coordinator, shutdown: SingleTrigger::new() })
    }
}

impl AsyncService for InferenceGrpcService {
    fn ident(self: Arc<Self>) -> &'static str {
        INFERENCE_GRPC_SERVICE
    }

    fn start(self: Arc<Self>) -> AsyncServiceFuture {
        Box::pin(async move {
            info!("{} starting", INFERENCE_GRPC_SERVICE);

            let addr: SocketAddr = self
                .listen_address
                .to_string()
                .parse()
                .map_err(|e| AsyncServiceError::Service(format!("Invalid inference gRPC listen address: {}", e)))?;

            let model_manager = self.coordinator.model_manager();
            let inference = seed_node::serving::inference::InferenceService::new(model_manager);

            let server = Server::builder().add_service(inference.into_server());
            let shutdown_signal = self.shutdown.listener.clone();

            let serve_future = server.serve(addr);
            tokio::select! {
                _ = shutdown_signal => {
                    info!("{} shutting down", INFERENCE_GRPC_SERVICE);
                    Ok(())
                }
                res = serve_future => {
                    match res {
                        Ok(()) => {
                            warn!("{} serve loop exited", INFERENCE_GRPC_SERVICE);
                            Ok(())
                        }
                        Err(e) => {
                            warn!("{} serve loop error: {}", INFERENCE_GRPC_SERVICE, e);
                            Err(AsyncServiceError::Service(format!("Inference gRPC server error: {}", e)))
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
            self.signal_exit();
            Ok(())
        })
    }
}
