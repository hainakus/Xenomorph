#![allow(dead_code)]

use anyhow::Result;
use seed_node::model::manager::ModelManager;
use seed_node::rpc::client::XenomorphRpcClient;
use seed_node::serving::inference::InferenceService;
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into())).init();

    info!("Starting Xenomorph Seed Node");

    // Initialize components
    let models_dir = std::env::var("XENO_MODELS_DIR").unwrap_or_else(|_| "/data/models".to_string());
    let node_rpc = std::env::var("XENO_NODE_RPC").unwrap_or_else(|_| "127.0.0.1:16110".to_string());
    let grpc_addr = std::env::var("XENO_GRPC_ADDR").unwrap_or_else(|_| "0.0.0.0:50051".to_string());

    let model_manager = Arc::new(ModelManager::new(models_dir).await?);
    let xenomorph_client = Arc::new(XenomorphRpcClient::new(&node_rpc).await?);

    // Initialize inference service
    let inference_service = InferenceService::new(model_manager.clone(), xenomorph_client.clone());

    // Start gRPC server
    let addr: SocketAddr = grpc_addr.parse()?;
    info!("gRPC server listening on {}", addr);

    Server::builder().add_service(inference_service.into_server()).serve(addr).await?;

    Ok(())
}
