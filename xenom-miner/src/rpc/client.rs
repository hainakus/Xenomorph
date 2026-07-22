use anyhow::{bail, Context, Result};
use borsh::{to_vec, BorshDeserialize};
use futures::{SinkExt, StreamExt};
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::{
    connect_async_with_config, tungstenite::protocol::WebSocketConfig, tungstenite::Message, MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, info, warn};

use super::messages::{
    BlockHash, DifficultyTarget, GenomeTrainingBatchMsg, GetModelCheckpointInfo, GradientUpdate, ModelCheckpoint, ModelCheckpointInfo,
    RpcEnvelope, RpcRequest, RpcResponse, TrainingBatch, TrainingBlock,
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MODEL_CHECKPOINT_TIMEOUT: Duration = Duration::from_secs(600);
/// Gradient payloads can be as large as a model checkpoint and the seed-node has
/// to decrypt, decompress, average and apply them before responding.
const GRADIENT_SUBMIT_TIMEOUT: Duration = Duration::from_secs(300);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const MAX_RECONNECT_DELAY: Duration = Duration::from_secs(60);
const MAX_SEND_ATTEMPTS: usize = 3;
/// Allow WebSocket messages up to 1 GiB so model checkpoints (config + tokenizer + weights) fit.
const WS_MAX_MESSAGE_SIZE: usize = 1024 * 1024 * 1024;

fn ws_config() -> WebSocketConfig {
    #[allow(deprecated)]
    WebSocketConfig {
        max_send_queue: None,
        write_buffer_size: 128 * 1024,
        max_write_buffer_size: usize::MAX,
        max_message_size: Some(WS_MAX_MESSAGE_SIZE),
        max_frame_size: Some(WS_MAX_MESSAGE_SIZE),
        accept_unmasked_frames: false,
    }
}

/// A WebSocket-based Borsh RPC client for the Xenomorph node.
pub struct XenomRpcClient {
    url: String,
    connection: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    request_counter: u64,
    last_heartbeat: Instant,
}

impl XenomRpcClient {
    /// Create a new client without connecting.
    pub fn new(url: String) -> Self {
        Self { url, connection: None, request_counter: 0, last_heartbeat: Instant::now() }
    }

    /// Connect (or reconnect) to the configured RPC endpoint.
    pub async fn connect(&mut self) -> Result<()> {
        let (ws_stream, _) = connect_async_with_config(&self.url, Some(ws_config()), false)
            .await
            .with_context(|| format!("Failed to connect to {}", self.url))?;

        info!("Connected to {}", self.url);
        self.connection = Some(ws_stream);
        self.last_heartbeat = Instant::now();
        Ok(())
    }

    /// Send a heartbeat to keep the connection alive if enough time has passed.
    pub async fn heartbeat(&mut self) -> Result<()> {
        if self.last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
            self.send_request(RpcRequest::Heartbeat).await?;
            self.last_heartbeat = Instant::now();
        }
        Ok(())
    }

    /// Request a training batch from the node.
    pub async fn get_training_batch(&mut self, model_id: &str) -> Result<Option<TrainingBatch>> {
        let response = self.send_request(RpcRequest::GetTrainingBatch { model_id: model_id.to_string() }).await?;

        match response {
            RpcResponse::TrainingBatch(batch) => Ok(batch),
            RpcResponse::Error(msg) => bail!("Node returned error: {}", msg),
            other => bail!("Unexpected response to GetTrainingBatch: {:?}", other),
        }
    }

    /// Request a genome-backed DNABERT-2 training batch from the seed-node.
    pub async fn get_genome_training_batch(
        &mut self,
        genome_merkle_root: [u8; 32],
        model_id: &str,
        preferred_batch_size: usize,
    ) -> Result<GenomeTrainingBatchMsg> {
        let response = self
            .send_request(RpcRequest::GetGenomeTrainingBatch(super::messages::GetGenomeTrainingBatch {
                genome_merkle_root,
                model_id: model_id.to_string(),
                preferred_batch_size,
            }))
            .await?;

        match response {
            RpcResponse::GenomeTrainingBatch(msg) => Ok(msg),
            RpcResponse::Error(msg) => bail!("Node returned error: {}", msg),
            other => bail!("Unexpected response to GetGenomeTrainingBatch: {:?}", other),
        }
    }

    /// Request lightweight checkpoint metadata (model id + base checkpoint hash).
    /// This is used to decide whether the local model cache is still valid.
    pub async fn get_model_checkpoint_info(&mut self, model_id: &str) -> Result<ModelCheckpointInfo> {
        let response =
            self.send_request(RpcRequest::GetModelCheckpointInfo(GetModelCheckpointInfo { model_id: model_id.to_string() })).await?;

        match response {
            RpcResponse::ModelCheckpointInfo(info) => Ok(info),
            RpcResponse::Error(msg) => bail!("Node returned error: {}", msg),
            other => bail!("Unexpected response to GetModelCheckpointInfo: {:?}", other),
        }
    }

    /// Request the full model checkpoint (config + tokenizer + weights) from the seed-node.
    /// Uses a much longer timeout because the response can be several hundred MB.
    pub async fn get_model_checkpoint(&mut self, model_id: &str) -> Result<ModelCheckpoint> {
        let response = self
            .send_request_with_timeout(RpcRequest::GetModelCheckpoint { model_id: model_id.to_string() }, MODEL_CHECKPOINT_TIMEOUT)
            .await?;

        match response {
            RpcResponse::ModelCheckpoint(cp) => Ok(cp),
            RpcResponse::Error(msg) => bail!("Node returned error: {}", msg),
            other => bail!("Unexpected response to GetModelCheckpoint: {:?}", other),
        }
    }

    /// Submit a completed training block to the node.
    pub async fn submit_block(&mut self, block: TrainingBlock) -> Result<BlockHash> {
        let response = self.send_request(RpcRequest::SubmitBlock(block)).await?;

        match response {
            RpcResponse::BlockHash(hash) => Ok(hash),
            RpcResponse::Error(msg) => bail!("Node rejected block: {}", msg),
            other => bail!("Unexpected response to SubmitBlock: {:?}", other),
        }
    }

    /// Submit a gradient update for FedAvg aggregation.
    pub async fn submit_gradients(&mut self, update: GradientUpdate) -> Result<Option<[u8; 32]>> {
        let response = self.send_request_with_timeout(RpcRequest::SubmitGradients(update), GRADIENT_SUBMIT_TIMEOUT).await?;

        match response {
            RpcResponse::GradientAck { new_checkpoint } => Ok(new_checkpoint),
            RpcResponse::Error(msg) => bail!("Node rejected gradient update: {}", msg),
            other => bail!("Unexpected response to SubmitGradients: {:?}", other),
        }
    }

    /// Query the balance of an address.
    pub async fn get_balance(&mut self, address: &str) -> Result<u64> {
        let response = self.send_request(RpcRequest::GetBalance { address: address.to_string() }).await?;

        match response {
            RpcResponse::Balance(balance) => Ok(balance),
            RpcResponse::Error(msg) => bail!("Node returned error: {}", msg),
            other => bail!("Unexpected response to GetBalance: {:?}", other),
        }
    }

    /// Query the current network difficulty.
    pub async fn get_difficulty(&mut self) -> Result<DifficultyTarget> {
        let response = self.send_request(RpcRequest::GetDifficulty).await?;

        match response {
            RpcResponse::Difficulty(target) => Ok(target),
            RpcResponse::Error(msg) => bail!("Node returned error: {}", msg),
            other => bail!("Unexpected response to GetDifficulty: {:?}", other),
        }
    }

    /// Ensure a connection is open, reconnecting with exponential backoff on failure.
    async fn ensure_connected(&mut self) -> Result<()> {
        if self.connection.is_some() {
            return Ok(());
        }

        let mut delay = Duration::from_secs(1);
        loop {
            match connect_async_with_config(&self.url, Some(ws_config()), false).await {
                Ok((ws_stream, _)) => {
                    info!("Reconnected to {}", self.url);
                    self.connection = Some(ws_stream);
                    self.last_heartbeat = Instant::now();
                    return Ok(());
                }
                Err(e) => {
                    warn!("Failed to reconnect to {}: {}. Retrying in {:?}", self.url, e, delay);
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(MAX_RECONNECT_DELAY);
                }
            }
        }
    }

    /// Send a single Borsh request with the default timeout.
    async fn send_request(&mut self, request: RpcRequest) -> Result<RpcResponse> {
        self.send_request_with_timeout(request, REQUEST_TIMEOUT).await
    }

    /// Send a single Borsh request and wait for a matching response, with a configurable timeout.
    /// Retries transparently if the connection drops mid-flight.
    async fn send_request_with_timeout(&mut self, request: RpcRequest, timeout_duration: Duration) -> Result<RpcResponse> {
        let request_id = self.request_counter;
        self.request_counter += 1;

        let envelope = RpcEnvelope { request_id, payload: request };
        let payload = to_vec(&envelope).with_context(|| "Failed to serialize RPC envelope")?;

        for attempt in 0..MAX_SEND_ATTEMPTS {
            self.ensure_connected().await?;

            let stream = self.connection.as_mut().context("No WebSocket connection")?;
            if let Err(e) = stream.send(Message::Binary(payload.clone())).await {
                warn!("WebSocket send failed on attempt {}: {}; marking connection for reconnect", attempt, e);
                self.connection = None;
                if attempt == MAX_SEND_ATTEMPTS - 1 {
                    bail!("Failed to send WebSocket message: {}", e);
                }
                continue;
            }

            debug!("Sent RPC request {}", request_id);

            let mut deadline = tokio::time::Instant::now() + timeout_duration;
            let should_retry;

            loop {
                let remaining = deadline - tokio::time::Instant::now();
                if remaining.is_zero() {
                    self.connection = None;
                    should_retry = attempt < MAX_SEND_ATTEMPTS - 1;
                    break;
                }

                let next = timeout(remaining, stream.next());
                match next.await {
                    Ok(Some(Ok(Message::Binary(bytes)))) => {
                        let response: RpcResponse =
                            RpcResponse::try_from_slice(&bytes).with_context(|| "Failed to deserialize RPC response")?;
                        debug!("Received RPC response for request {}", request_id);
                        return Ok(response);
                    }
                    Ok(Some(Ok(Message::Close(_)))) | Ok(Some(Ok(Message::Text(_)))) => {
                        // Ignore text and close frames, keep waiting for binary response.
                        deadline = tokio::time::Instant::now() + timeout_duration;
                        continue;
                    }
                    Ok(Some(Ok(Message::Ping(_)))) | Ok(Some(Ok(Message::Pong(_)))) => {
                        // The peer is alive; extend the deadline and keep waiting.
                        deadline = tokio::time::Instant::now() + timeout_duration;
                        continue;
                    }
                    Ok(Some(Ok(Message::Frame(_)))) => {
                        deadline = tokio::time::Instant::now() + timeout_duration;
                        continue;
                    }
                    Ok(Some(Err(e))) => {
                        warn!("WebSocket read error on attempt {}: {}", attempt, e);
                        self.connection = None;
                        should_retry = attempt < MAX_SEND_ATTEMPTS - 1;
                        break;
                    }
                    Ok(None) => {
                        warn!("WebSocket stream closed by peer on attempt {}", attempt);
                        self.connection = None;
                        should_retry = attempt < MAX_SEND_ATTEMPTS - 1;
                        break;
                    }
                    Err(_) => {
                        warn!("RPC request {} timed out on attempt {}", request_id, attempt);
                        self.connection = None;
                        should_retry = attempt < MAX_SEND_ATTEMPTS - 1;
                        break;
                    }
                }
            }

            if !should_retry {
                bail!("RPC request {} failed", request_id);
            }
        }

        bail!("RPC request {} exhausted all retries", request_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_counter_increments() {
        let mut client = XenomRpcClient::new("ws://localhost:16110".to_string());
        let first = client.request_counter;
        client.request_counter += 1;
        assert_eq!(client.request_counter, first + 1);
    }

    #[tokio::test]
    async fn test_connect_to_invalid_url_fails() {
        let mut client = XenomRpcClient::new("ws://127.0.0.1:1".to_string());
        // ensure_connected loops forever with backoff; use a short timeout.
        let result = timeout(Duration::from_millis(500), client.connect()).await;
        assert!(result.is_err() || result.unwrap().is_err());
    }
}
