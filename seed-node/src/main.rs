#![allow(dead_code)]

use anyhow::Result;
use kaspa_consensus_core::network::NetworkType;
use seed_node::genome::GenomeStorage;
use seed_node::model::manager::ModelManager;
use seed_node::model::storage::ModelStorage;
use seed_node::p2p::P2pGossipHandle;
use seed_node::rpc::client::XenomorphRpcClient;
use seed_node::serving::inference::InferenceService;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into())).init();

    info!("Starting Xenomorph Seed Node");

    // Initialize components
    let models_dir = std::env::var("XENO_MODELS_DIR").unwrap_or_else(|_| "/data/models".to_string());
    let node_rpc = std::env::var("XENO_NODE_RPC").unwrap_or_else(|_| "127.0.0.1:16110".to_string());
    let node_p2p = std::env::var("XENO_NODE_P2P").unwrap_or_else(|_| "127.0.0.1:16111".to_string());
    let grpc_addr = std::env::var("XENO_GRPC_ADDR").unwrap_or_else(|_| "0.0.0.0:50051".to_string());
    let miner_ws_addr = std::env::var("XENO_MINER_WS_ADDR").unwrap_or_else(|_| "0.0.0.0:17110".to_string());
    let seed_host = std::env::var("XENO_SEED_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let default_model_id = std::env::var("XENO_DEFAULT_MODEL_ID").unwrap_or_else(|_| "multimolecule/dnabert2".to_string());
    let network_str = std::env::var("XENO_NETWORK").unwrap_or_else(|_| "devnet".to_string());
    let network_type = NetworkType::from_str(&network_str).unwrap_or(NetworkType::Devnet);

    let model_key = ModelStorage::derive_encryption_key();
    let model_manager = Arc::new(ModelManager::new_with_key(models_dir.clone(), model_key).await?);
    let genome_storage = Arc::new(RwLock::new(GenomeStorage::new(PathBuf::from(models_dir).join("genomes")).await?));

    // The seed-node is only considered ready once the default model is available.
    // Block startup until the model is downloaded and stored locally.
    info!("Ensuring default model {} is available...", default_model_id);
    model_manager.ensure_model_downloaded(&default_model_id).await?;
    info!("Default model {} is ready", default_model_id);

    // Join the P2P gossip network and announce the active checkpoint.
    let p2p_gossip = match P2pGossipHandle::connect(node_p2p, network_type).await {
        Ok(gossip) => {
            if let Some(model_info) = model_manager.get_model(&default_model_id).await {
                let weights_hash = model_info.checkpoint.weights_hash;
                let listen_addr = announce_listen_addr(&miner_ws_addr, &seed_host);
                gossip.announce(default_model_id.clone(), weights_hash, weights_hash, listen_addr).await;
                info!("Announced checkpoint {} on P2P gossip", hex::encode(weights_hash));
            }
            Some(gossip)
        }
        Err(e) => {
            warn!("P2P gossip disabled: {e}");
            None
        }
    };

    let xenomorph_client = Arc::new(XenomorphRpcClient::new(&node_rpc).await?);

    // Initialize inference service
    let inference_service = InferenceService::new(model_manager.clone(), xenomorph_client.clone());

    // Start gRPC server in the background
    let addr: SocketAddr = grpc_addr.parse()?;
    info!("gRPC server listening on {}", addr);
    let grpc_handle = tokio::spawn(async move { Server::builder().add_service(inference_service.into_server()).serve(addr).await });

    // Start miner WebSocket server in the foreground
    let _p2p_gossip = p2p_gossip;
    seed_node::rpc::server::run_miner_server(&miner_ws_addr, model_manager, genome_storage, Some(xenomorph_client.clone())).await?;

    // If the WebSocket server exits, wait for gRPC too
    grpc_handle.await??;

    Ok(())
}

fn announce_listen_addr(miner_ws_addr: &str, seed_host: &str) -> Option<SocketAddr> {
    let addr: SocketAddr = miner_ws_addr.parse().ok()?;
    let ip = if addr.ip().is_unspecified() { seed_host.parse().unwrap_or(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))) } else { addr.ip() };
    Some(SocketAddr::new(ip, addr.port()))
}
