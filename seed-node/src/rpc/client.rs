use crate::rpc::borsh_codec::BorshCodec;
use anyhow::{anyhow, Result};
use borsh::{BorshDeserialize, BorshSerialize};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::instrument;

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

pub struct XenomorphRpcClient {
    addr: SocketAddr,
    codec: BorshCodec,
}

impl XenomorphRpcClient {
    pub async fn new(addr: &str) -> Result<Self> {
        let socket_addr = tokio::net::lookup_host(addr)
            .await
            .map_err(|e| anyhow!("Failed to resolve {}: {}", addr, e))?
            .next()
            .ok_or_else(|| anyhow!("No addresses found for {}", addr))?;

        Ok(Self { addr: socket_addr, codec: BorshCodec::new() })
    }

    #[instrument(skip(self))]
    pub async fn get_model_checkpoint(&self, model_id: &str, version: u32) -> Result<GetModelCheckpointResponse> {
        let request = GetModelCheckpointRequest { model_id: model_id.to_string(), version };

        let message = RpcMessage::GetModelCheckpoint(request);
        let response = self.send_message(message).await?;

        match response {
            RpcMessage::GetModelCheckpointResponse(resp) => Ok(resp),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    #[instrument(skip(self, training_proof))]
    pub async fn submit_training_block(
        &self,
        model_id: &str,
        training_proof: Vec<u8>,
        block_height: u64,
    ) -> Result<SubmitTrainingBlockResponse> {
        let request = SubmitTrainingBlockRequest { model_id: model_id.to_string(), training_proof, block_height };

        let message = RpcMessage::SubmitTrainingBlock(request);
        let response = self.send_message(message).await?;

        match response {
            RpcMessage::SubmitTrainingBlockResponse(resp) => Ok(resp),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    #[instrument(skip(self))]
    pub async fn ping(&self) -> Result<()> {
        let message = RpcMessage::Ping;
        let response = self.send_message(message).await?;

        match response {
            RpcMessage::Pong => Ok(()),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    async fn send_message(&self, message: RpcMessage) -> Result<RpcMessage> {
        let mut stream = TcpStream::connect(self.addr).await.map_err(|e| anyhow!("Failed to connect: {}", e))?;

        // Serialize and send message
        let serialized = self.codec.serialize(&message)?;
        let length = serialized.len() as u32;

        stream.write_all(&length.to_le_bytes()).await.map_err(|e| anyhow!("Failed to write length: {}", e))?;
        stream.write_all(&serialized).await.map_err(|e| anyhow!("Failed to write data: {}", e))?;

        // Read response length
        let mut length_buf = [0u8; 4];
        stream.read_exact(&mut length_buf).await.map_err(|e| anyhow!("Failed to read length: {}", e))?;
        let response_length = u32::from_le_bytes(length_buf) as usize;

        // Read response data
        let mut response_buf = vec![0u8; response_length];
        stream.read_exact(&mut response_buf).await.map_err(|e| anyhow!("Failed to read data: {}", e))?;

        // Deserialize response
        let response = self.codec.deserialize(&response_buf)?;

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let request = GetModelCheckpointRequest { model_id: "test_model".to_string(), version: 1 };

        let codec = BorshCodec::new();
        let serialized = codec.serialize(&request).unwrap();
        let deserialized = codec.deserialize::<GetModelCheckpointRequest>(&serialized).unwrap();

        assert_eq!(request.model_id, deserialized.model_id);
        assert_eq!(request.version, deserialized.version);
    }
}
