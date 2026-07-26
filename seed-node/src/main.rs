#![allow(dead_code)]

use anyhow::Result;
use kaspa_consensus_core::network::NetworkType;
use seed_node::genome::GenomeStorage;
use seed_node::model::manager::ModelManager;
use seed_node::model::storage::ModelStorage;
use seed_node::p2p::P2pGossipHandle;
use seed_node::quic::ModelFileProvider;
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
use xenom_miner::lora::LoraConfig;
use xenom_quic::CheckpointTransferServer;

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
    let quic_listen = std::env::var("XENO_QUIC_LISTEN").unwrap_or_else(|_| "0.0.0.0:17111".to_string());
    let quic_external = std::env::var("XENO_QUIC_EXTERNAL").ok();
    let quic_max_transfers: u32 = std::env::var("XENO_QUIC_MAX_TRANSFERS").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let seed_host = std::env::var("XENO_SEED_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let default_model_id = std::env::var("XENO_DEFAULT_MODEL_ID").unwrap_or_else(|_| "multimolecule/dnabert2".to_string());
    let network_str = std::env::var("XENO_NETWORK").unwrap_or_else(|_| "devnet".to_string());
    let network_type = NetworkType::from_str(&network_str).unwrap_or(NetworkType::Devnet);

    let model_key = ModelStorage::derive_encryption_key();
    let lora_config = LoraConfig::from_env();
    let model_manager = Arc::new(ModelManager::new_with_key(models_dir.clone(), model_key, lora_config).await?);
    let genome_storage = Arc::new(RwLock::new(GenomeStorage::new(PathBuf::from(models_dir.clone()).join("genomes")).await?));

    // The seed-node is only considered ready once the default model is available.
    // Block startup until the model is downloaded, stored locally and loaded into memory.
    info!("Ensuring default model {} is available...", default_model_id);
    model_manager.ensure_model_downloaded(&default_model_id).await?;
    model_manager.load_model(&default_model_id).await?;
    info!("Default model {} is ready", default_model_id);

    // Start the QUIC bulk transfer server and expose the model files to peers.
    let bind_addr: SocketAddr = quic_listen.parse()?;
    let provider = Arc::new(ModelFileProvider::new(model_manager.clone()));
    let quic_server = CheckpointTransferServer::with_max_transfers(bind_addr, provider, quic_max_transfers as usize).await?;
    let quic_local_addr = quic_server.local_addr()?;
    info!("QUIC transfer server listening on {}", quic_local_addr);

    // Compute the address to announce in P2P gossip (external override, otherwise bind address
    // with any unspecified IP resolved via XENO_SEED_HOST).
    let quic_announce_addr = quic_announce_addr(quic_local_addr, quic_external.as_deref(), &seed_host);

    // Hold the QUIC server alive for the lifetime of the process.
    tokio::spawn(async move {
        let _ = quic_server;
        std::future::pending::<()>().await
    });

    // Join the P2P gossip network and announce the active checkpoint.
    let gossip_key_path = PathBuf::from(&models_dir).join("gossip.key");
    let p2p_gossip = match P2pGossipHandle::connect(node_p2p, network_type, &gossip_key_path).await {
        Ok(gossip) => {
            if let Some(model_info) = model_manager.get_model(&default_model_id).await {
                let weights_hash = model_info.checkpoint.weights_hash;
                gossip.announce(default_model_id.clone(), weights_hash, weights_hash, quic_announce_addr).await;
                info!("Announced checkpoint {} on P2P gossip (listen_addr: {:?})", hex::encode(weights_hash), quic_announce_addr);
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
    let inference_service = InferenceService::new(model_manager.clone());

    // Start gRPC server in the background
    let addr: SocketAddr = grpc_addr.parse()?;
    info!("gRPC server listening on {}", addr);
    let grpc_handle = tokio::spawn(async move { Server::builder().add_service(inference_service.into_server()).serve(addr).await });

    // Start miner WebSocket server in the foreground
    seed_node::rpc::server::run_miner_server(
        &miner_ws_addr,
        model_manager,
        genome_storage,
        Some(xenomorph_client.clone()),
        p2p_gossip,
    )
    .await?;

    // If the WebSocket server exits, wait for gRPC too
    grpc_handle.await??;

    Ok(())
}

fn quic_announce_addr(local_addr: SocketAddr, external: Option<&str>, seed_host: &str) -> Option<SocketAddr> {
    if let Some(external) = external {
        return external.parse().ok();
    }

    let ip = if local_addr.ip().is_unspecified() {
        seed_host.parse().ok().or_else(|| local_ip_address::local_ip().ok()).unwrap_or(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)))
    } else {
        local_addr.ip()
    };
    Some(SocketAddr::new(ip, local_addr.port()))
}
