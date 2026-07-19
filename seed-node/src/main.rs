#![allow(dead_code)]

use anyhow::Result;
use seed_node::genome::GenomeStorage;
use seed_node::model::manager::ModelManager;
use seed_node::rpc::client::XenomorphRpcClient;
use seed_node::serving::inference::InferenceService;
use sha2::{Digest, Sha256};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// Derive a 32-byte AES key for model-file encryption.
///
/// If `XENO_MODEL_KEY` is a 64-character hex string it is decoded directly;
/// otherwise the value (or a default devnet string) is hashed with SHA-256.
/// The key must stay the same across restarts, otherwise stored models cannot
/// be decrypted and will be re-downloaded.
fn derive_model_encryption_key() -> [u8; 32] {
    const DEFAULT_KEY: &str = "xenom-devnet-model-key";
    let seed = std::env::var("XENO_MODEL_KEY").unwrap_or_else(|_| DEFAULT_KEY.to_string());
    let seed = seed.trim();

    if seed.len() == 64 {
        if let Ok(decoded) = hex::decode(seed) {
            if decoded.len() == 32 {
                let mut key = [0u8; 32];
                key.copy_from_slice(&decoded);
                return key;
            }
        }
    }

    if seed == DEFAULT_KEY {
        warn!("XENO_MODEL_KEY not set; using default devnet key. In production set XENO_MODEL_KEY to a strong secret.");
    } else {
        info!("Deriving model encryption key from XENO_MODEL_KEY");
    }

    let hash = Sha256::digest(seed.as_bytes());
    let mut key = [0u8; 32];
    key.copy_from_slice(&hash);
    key
}

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

    let model_key = derive_model_encryption_key();
    let model_manager = Arc::new(ModelManager::new_with_key(models_dir.clone(), model_key).await?);
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
