#![allow(dead_code)]

use anyhow::Result;
use seed_node::genome::GenomeStorage;
use seed_node::model::manager::ModelManager;
use seed_node::rpc::client::XenomorphRpcClient;
use seed_node::serving::inference::InferenceService;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
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
    let miner_ws_addr = std::env::var("XENO_MINER_WS_ADDR").unwrap_or_else(|_| "0.0.0.0:17110".to_string());
    let default_model_id = std::env::var("XENO_DEFAULT_MODEL_ID").unwrap_or_else(|_| "multimolecule/dnabert2".to_string());

    let model_manager = Arc::new(ModelManager::new(models_dir.clone()).await?);
    let genome_storage = Arc::new(RwLock::new(GenomeStorage::new(PathBuf::from(models_dir).join("genomes")).await?));

    // The seed-node is only considered ready once the default model is available.
    // Block startup until the model is downloaded and stored locally.
    info!("Ensuring default model {} is available...", default_model_id);
    model_manager.ensure_model_downloaded(&default_model_id).await?;
    info!("Default model {} is ready", default_model_id);

    let xenomorph_client = Arc::new(XenomorphRpcClient::new(&node_rpc).await?);

    // Initialize inference service
    let inference_service = InferenceService::new(model_manager.clone(), xenomorph_client.clone());

    // Start gRPC server in the background
    let addr: SocketAddr = grpc_addr.parse()?;
    info!("gRPC server listening on {}", addr);
    let grpc_handle = tokio::spawn(async move {
        Server::builder().add_service(inference_service.into_server()).serve(addr).await
    });

    // Start miner WebSocket server in the foreground
    seed_node::rpc::server::run_miner_server(&miner_ws_addr, model_manager, genome_storage).await?;

    // If the WebSocket server exits, wait for gRPC too
    grpc_handle.await??;

    Ok(())
}
