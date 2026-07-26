use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use borsh::{to_vec, BorshDeserialize};
use quinn::{Endpoint, ServerConfig};
use tracing::{info, warn};

use crate::cert::generate_self_signed_cert;
use crate::protocol::{CheckpointFileRequest, CheckpointFileResponseHeader, ResponseStatus};

/// Function that resolves a file request to its bytes, or `None` when the file is not found.
pub type FileProvider = Arc<dyn Fn(&CheckpointFileRequest) -> Result<Option<Vec<u8>>> + Send + Sync>;

/// QUIC server that serves checkpoint files to connected miners.
pub struct CheckpointTransferServer {
    endpoint: Endpoint,
}

impl CheckpointTransferServer {
    /// Bind a new QUIC transfer server on `bind_addr`.
    pub async fn new(bind_addr: SocketAddr, file_provider: FileProvider) -> Result<Self> {
        let (cert, key) =
            generate_self_signed_cert(&["localhost".to_string()]).context("failed to generate QUIC server certificate")?;
        let server_config = ServerConfig::with_single_cert(vec![cert], key).context("failed to create QUIC server config")?;
        let endpoint = Endpoint::server(server_config, bind_addr).context("failed to bind QUIC server endpoint")?;
        let local_addr = endpoint.local_addr()?;
        info!("QUIC transfer server listening on {}", local_addr);

        tokio::spawn(run_server(endpoint.clone(), file_provider));

        Ok(Self { endpoint })
    }

    /// Return the local address this server bound to.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint.local_addr().context("failed to get QUIC server local address")
    }
}

async fn run_server(endpoint: Endpoint, file_provider: FileProvider) {
    while let Some(incoming) = endpoint.accept().await {
        let file_provider = file_provider.clone();
        tokio::spawn(async move {
            match incoming.accept() {
                Ok(conn) => {
                    if let Err(e) = handle_connection(conn, file_provider).await {
                        warn!("QUIC connection error: {e}");
                    }
                }
                Err(e) => warn!("QUIC incoming connection rejected: {e}"),
            }
        });
    }
}

async fn handle_connection(conn: quinn::Connecting, file_provider: FileProvider) -> Result<()> {
    let conn = conn.await.context("incoming QUIC connection failed")?;
    while let Ok((send, recv)) = conn.accept_bi().await {
        let file_provider = file_provider.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_stream(send, recv, file_provider).await {
                warn!("QUIC stream error: {e}");
            }
        });
    }
    Ok(())
}

async fn handle_stream(mut send: quinn::SendStream, mut recv: quinn::RecvStream, file_provider: FileProvider) -> Result<()> {
    let mut len_bytes = [0u8; 4];
    recv.read_exact(&mut len_bytes).await.context("failed to read request length")?;
    let len = u32::from_le_bytes(len_bytes) as usize;

    let mut req_buf = vec![0u8; len];
    recv.read_exact(&mut req_buf).await.context("failed to read request body")?;
    let request = CheckpointFileRequest::try_from_slice(&req_buf).context("failed to deserialize request")?;

    let (status, payload) = match file_provider(&request) {
        Ok(Some(bytes)) => (ResponseStatus::Ok, bytes),
        Ok(None) => (ResponseStatus::NotFound, Vec::new()),
        Err(e) => {
            warn!("file provider error for {}: {e}", request.model_id);
            (ResponseStatus::NotAuthorized, Vec::new())
        }
    };

    let header = CheckpointFileResponseHeader { status, length: payload.len() as u64 };
    let header_bytes = to_vec(&header).context("failed to serialize response header")?;
    send.write_all(&(header_bytes.len() as u32).to_le_bytes()).await.context("failed to write header length")?;
    send.write_all(&header_bytes).await.context("failed to write response header")?;
    if !payload.is_empty() {
        send.write_all(&payload).await.context("failed to write response payload")?;
    }
    send.finish().context("failed to finish send stream")?;

    Ok(())
}
