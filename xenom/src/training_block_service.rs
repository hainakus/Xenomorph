//! Borsh training-block RPC service.
//!
//! This service accepts `SubmitTrainingBlock` requests from the Xenomorph seed-node,
//! builds a Kaspa block template with the training proof committed in the coinbase
//! extra-data, solves the block PoW, and submits the block to the local consensus.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{anyhow, Result as AnyhowResult};
use borsh::{to_vec, BorshDeserialize, BorshSerialize};
use kaspa_addresses::{Address, Prefix};
use kaspa_consensus_core::header::Header;
use kaspa_consensus_core::network::NetworkType;
use kaspa_core::task::service::{AsyncService, AsyncServiceError, AsyncServiceFuture};
use kaspa_core::{info, trace, warn};
use kaspa_hashes::Hash;
use kaspa_pow::State as PowState;
use kaspa_rpc_core::api::rpc::RpcApi;
use kaspa_rpc_core::{GetBlockTemplateRequest, SubmitBlockReport, SubmitBlockRequest};
use kaspa_rpc_service::service::RpcCoreService;
use kaspa_utils::networking::ContextualNetAddress;
use kaspa_utils::triggers::SingleTrigger;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;
const TRAINING_BLOCK_SERVICE: &str = "training-block-rpc";

/// Raw training proof as serialized by the `xenom-miner` / `seed-node`.
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct TrainingProof {
    pub base_checkpoint: [u8; 32],
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: [u8; 32],
    pub zk_proof: Vec<u8>,
    pub batch_indices: Vec<u64>,
    pub compute_time_ms: u64,
}

/// Compact data embedded in the coinbase extra-data payload.
#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct CoinbaseExtraData {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: [u8; 32],
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct GetModelCheckpointRequest {
    pub model_id: String,
    pub version: u32,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct GetModelCheckpointResponse {
    pub checkpoint_data: Vec<u8>,
    pub block_height: u64,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct SubmitTrainingBlockRequest {
    pub model_id: String,
    pub miner_address: String,
    pub training_proof: Vec<u8>,
    pub block_height: u64,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct SubmitTrainingBlockResponse {
    pub accepted: bool,
    pub block_hash: [u8; 32],
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub enum RpcMessage {
    GetModelCheckpoint(GetModelCheckpointRequest),
    GetModelCheckpointResponse(GetModelCheckpointResponse),
    SubmitTrainingBlock(SubmitTrainingBlockRequest),
    SubmitTrainingBlockResponse(SubmitTrainingBlockResponse),
    Ping,
    Pong,
}

pub struct TrainingBlockService {
    listen_address: ContextualNetAddress,
    network_type: NetworkType,
    rpc_core_service: Arc<RpcCoreService>,
    shutdown: SingleTrigger,
}

impl TrainingBlockService {
    pub fn new(listen_address: ContextualNetAddress, network_type: NetworkType, rpc_core_service: Arc<RpcCoreService>) -> Arc<Self> {
        Arc::new(Self { listen_address, network_type, rpc_core_service, shutdown: SingleTrigger::new() })
    }

    async fn serve(self: Arc<Self>, listener: TcpListener) -> AnyhowResult<()> {
        loop {
            let (stream, peer) = listener.accept().await?;
            let service = self.clone();
            tokio::spawn(async move {
                if let Err(e) = service.handle_connection(stream, peer).await {
                    warn!("Training block connection from {} error: {}", peer, e);
                }
            });
        }
    }

    async fn handle_connection(self: Arc<Self>, mut stream: TcpStream, peer: SocketAddr) -> AnyhowResult<()> {
        trace!("Training block connection from {}", peer);

        let mut length_buf = [0u8; 4];
        stream.read_exact(&mut length_buf).await?;
        let length = u32::from_le_bytes(length_buf) as usize;
        if length > MAX_MESSAGE_SIZE {
            return Err(anyhow!("Training block message too large: {} bytes", length));
        }

        let mut buf = vec![0u8; length];
        stream.read_exact(&mut buf).await?;
        let request = RpcMessage::try_from_slice(&buf).map_err(|e| anyhow!("Failed to deserialize training block request: {}", e))?;

        let response = self.handle_message(request).await;
        let response_bytes = to_vec(&response).map_err(|e| anyhow!("Failed to serialize training block response: {}", e))?;
        let response_len = response_bytes.len() as u32;
        stream.write_all(&response_len.to_le_bytes()).await?;
        stream.write_all(&response_bytes).await?;
        Ok(())
    }

    async fn handle_message(&self, message: RpcMessage) -> RpcMessage {
        match message {
            RpcMessage::SubmitTrainingBlock(request) => self.submit_training_block(request).await,
            RpcMessage::Ping => RpcMessage::Pong,
            _ => RpcMessage::SubmitTrainingBlockResponse(SubmitTrainingBlockResponse { accepted: false, block_hash: [0u8; 32] }),
        }
    }

    async fn submit_training_block(&self, request: SubmitTrainingBlockRequest) -> RpcMessage {
        // Validate that the miner address belongs to the network we are running.
        let address = match Address::try_from(request.miner_address.as_str()) {
            Ok(addr) => addr,
            Err(e) => {
                warn!("Rejecting training block: invalid miner address {}: {}", request.miner_address, e);
                return error_response(format!("Invalid miner address: {}", e));
            }
        };
        let expected_prefix = Prefix::from(self.network_type);
        if address.prefix != expected_prefix {
            warn!(
                "Rejecting training block: address prefix {} does not match network {:?} (expected {})",
                address.prefix, self.network_type, expected_prefix
            );
            return error_response(format!(
                "Address prefix {} does not match network {:?} (expected {})",
                address.prefix, self.network_type, expected_prefix
            ));
        }

        // Deserialize the training proof produced by the miner.
        let proof = match TrainingProof::try_from_slice(&request.training_proof) {
            Ok(proof) => proof,
            Err(e) => {
                warn!("Rejecting training block: invalid training proof: {}", e);
                return error_response(format!("Invalid training proof: {}", e));
            }
        };

        // Embed a compact proof summary into the coinbase extra-data.
        let extra_data = CoinbaseExtraData {
            model_id: request.model_id,
            base_checkpoint: proof.base_checkpoint,
            loss_before: proof.loss_before,
            loss_after: proof.loss_after,
            gradients_commitment: proof.gradients_commitment,
        };
        let extra_data_bytes = match to_vec(&extra_data) {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!("Failed to serialize coinbase extra data: {}", e);
                return error_response(format!("Failed to serialize coinbase extra data: {}", e));
            }
        };

        // Ask the local consensus for a block template paying the miner.
        let template_request = GetBlockTemplateRequest::new(address, extra_data_bytes);
        let template = match self.rpc_core_service.get_block_template_call(None, template_request).await {
            Ok(template) => template,
            Err(e) => {
                warn!("get_block_template failed for training block: {}", e);
                return error_response(format!("get_block_template failed: {}", e));
            }
        };

        if !template.is_synced {
            warn!("Node is not synced; cannot produce training block");
            return error_response("Node is not synced".to_string());
        }

        // Solve the block PoW.
        let mut raw_block = template.block;
        let mut header: Header = raw_block.header.into();
        let state = PowState::new(&header);

        let mut nonce = 0u64;
        let found_nonce = loop {
            if state.check_pow(nonce).0 {
                break nonce;
            }
            nonce = nonce.wrapping_add(1);
            if nonce == 0 {
                warn!("Failed to solve PoW for training block");
                return error_response("Failed to solve block PoW".to_string());
            }
        };

        header.nonce = found_nonce;
        header.finalize();
        let block_hash: [u8; 32] = header.hash.as_bytes();
        raw_block.header = header.into();

        // Submit the solved block to the consensus layer.
        let submit_request = SubmitBlockRequest { block: raw_block, allow_non_daa_blocks: false };
        let submit_response = match self.rpc_core_service.submit_block_call(None, submit_request).await {
            Ok(response) => response,
            Err(e) => {
                warn!("submit_block failed for training block: {}", e);
                return error_response(format!("submit_block failed: {}", e));
            }
        };

        let accepted = matches!(submit_response.report, SubmitBlockReport::Success);
        if accepted {
            info!("Accepted training block {} (hash {})", request.block_height, Hash::from(block_hash));
        } else {
            warn!("Consensus rejected training block {}: {:?}", request.block_height, submit_response.report);
        }

        RpcMessage::SubmitTrainingBlockResponse(SubmitTrainingBlockResponse { accepted, block_hash })
    }
}

fn error_response(_error: String) -> RpcMessage {
    RpcMessage::SubmitTrainingBlockResponse(SubmitTrainingBlockResponse { accepted: false, block_hash: [0u8; 32] })
}

impl AsyncService for TrainingBlockService {
    fn ident(self: Arc<Self>) -> &'static str {
        TRAINING_BLOCK_SERVICE
    }

    fn start(self: Arc<Self>) -> AsyncServiceFuture {
        Box::pin(async move {
            let listen = self.listen_address.to_string();
            let listener = TcpListener::bind(&listen)
                .await
                .map_err(|e| AsyncServiceError::Service(format!("Failed to bind training block RPC {}: {}", listen, e)))?;
            info!("{} listening on {}", TRAINING_BLOCK_SERVICE, listen);

            let shutdown_signal = self.shutdown.listener.clone();
            let service = self.clone();

            tokio::select! {
                res = service.serve(listener) => {
                    if let Err(e) = res {
                        warn!("{} serve loop exited: {}", TRAINING_BLOCK_SERVICE, e);
                    }
                    Ok(())
                }
                _ = shutdown_signal => {
                    info!("{} shutting down", TRAINING_BLOCK_SERVICE);
                    Ok(())
                }
            }
        })
    }

    fn signal_exit(self: Arc<Self>) {
        self.shutdown.trigger.trigger();
    }

    fn stop(self: Arc<Self>) -> AsyncServiceFuture {
        Box::pin(async move {
            trace!("{} stopped", TRAINING_BLOCK_SERVICE);
            Ok(())
        })
    }
}
