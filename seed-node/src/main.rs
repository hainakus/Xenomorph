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
    let model_manager = Arc::new(ModelManager::new("/data/models".to_string()).await?);
    let xenomorph_client = Arc::new(XenomorphRpcClient::new("127.0.0.1:16110").await?);

    // Initialize inference service
    let inference_service = InferenceService::new(model_manager.clone(), xenomorph_client.clone());

    // Start gRPC server
    let addr: SocketAddr = "0.0.0.0:50051".parse()?;
    info!("gRPC server listening on {}", addr);

    Server::builder().add_service(inference_service.into_server()).serve(addr).await?;

    Ok(())
}
