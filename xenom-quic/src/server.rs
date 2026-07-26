use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use borsh::{to_vec, BorshDeserialize};
use quinn::{Endpoint, ServerConfig};
use tokio::sync::Semaphore;
use tracing::{info, warn};

use crate::cert::generate_self_signed_cert;
use crate::protocol::{CheckpointFileRequest, CheckpointFileResponseHeader, ResponseStatus};

/// Default maximum number of concurrent checkpoint file transfers.
pub const DEFAULT_MAX_CONCURRENT_TRANSFERS: usize = 64;

/// Trait for resolving a `CheckpointFileRequest` to its encrypted bytes and
/// content hash. Returning `Ok(None)` means the file was not found.
#[async_trait]
pub trait FileProvider: Send + Sync {
    async fn get_file(&self, request: &CheckpointFileRequest) -> Result<Option<(Vec<u8>, [u8; 32])>>;
}

/// QUIC server that serves checkpoint files to connected miners.
pub struct CheckpointTransferServer {
    endpoint: Endpoint,
}

impl CheckpointTransferServer {
    /// Bind a new QUIC transfer server on `bind_addr` using the default
    /// concurrency limit.
    pub async fn new(bind_addr: SocketAddr, file_provider: Arc<dyn FileProvider>) -> Result<Self> {
        Self::with_max_transfers(bind_addr, file_provider, DEFAULT_MAX_CONCURRENT_TRANSFERS).await
    }

    /// Bind a new QUIC transfer server with a custom concurrency limit.
    pub async fn with_max_transfers(
        bind_addr: SocketAddr,
        file_provider: Arc<dyn FileProvider>,
        max_transfers: usize,
    ) -> Result<Self> {
        let (cert, key) =
            generate_self_signed_cert(&["localhost".to_string()]).context("failed to generate QUIC server certificate")?;
        let server_config = ServerConfig::with_single_cert(vec![cert], key).context("failed to create QUIC server config")?;
        let endpoint = Endpoint::server(server_config, bind_addr).context("failed to bind QUIC server endpoint")?;
        let local_addr = endpoint.local_addr()?;
        info!("QUIC transfer server listening on {} (max transfers: {})", local_addr, max_transfers);

        let semaphore = Arc::new(Semaphore::new(max_transfers));
        tokio::spawn(run_server(endpoint.clone(), file_provider, semaphore));

        Ok(Self { endpoint })
    }

    /// Return the local address this server bound to.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint.local_addr().context("failed to get QUIC server local address")
    }
}

async fn run_server(endpoint: Endpoint, file_provider: Arc<dyn FileProvider>, semaphore: Arc<Semaphore>) {
    while let Some(incoming) = endpoint.accept().await {
        let file_provider = file_provider.clone();
        let semaphore = semaphore.clone();
        tokio::spawn(async move {
            match incoming.accept() {
                Ok(conn) => {
                    if let Err(e) = handle_connection(conn, file_provider, semaphore).await {
                        warn!("QUIC connection error: {e}");
                    }
                }
                Err(e) => warn!("QUIC incoming connection rejected: {e}"),
            }
        });
    }
}

async fn handle_connection(conn: quinn::Connecting, file_provider: Arc<dyn FileProvider>, semaphore: Arc<Semaphore>) -> Result<()> {
    let conn = conn.await.context("incoming QUIC connection failed")?;
    while let Ok((send, recv)) = conn.accept_bi().await {
        let file_provider = file_provider.clone();
        let semaphore = semaphore.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_stream(send, recv, file_provider, semaphore).await {
                warn!("QUIC stream error: {e}");
            }
        });
    }
    Ok(())
}

async fn handle_stream(
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    file_provider: Arc<dyn FileProvider>,
    semaphore: Arc<Semaphore>,
) -> Result<()> {
    let _permit = semaphore.acquire().await.context("failed to acquire transfer semaphore permit")?;

    let mut len_bytes = [0u8; 4];
    recv.read_exact(&mut len_bytes).await.context("failed to read request length")?;
    let len = u32::from_le_bytes(len_bytes) as usize;

    let mut req_buf = vec![0u8; len];
    recv.read_exact(&mut req_buf).await.context("failed to read request body")?;
    let request = CheckpointFileRequest::try_from_slice(&req_buf).context("failed to deserialize request")?;

    let (status, payload, content_hash) = match file_provider.get_file(&request).await {
        Ok(Some((bytes, hash))) => (ResponseStatus::Ok, bytes, hash),
        Ok(None) => (ResponseStatus::NotFound, Vec::new(), [0u8; 32]),
        Err(e) => {
            warn!("file provider error for {}: {e}", request.model_id);
            (ResponseStatus::NotAuthorized, Vec::new(), [0u8; 32])
        }
    };

    let header = CheckpointFileResponseHeader { status, length: payload.len() as u64, content_hash };
    let header_bytes = to_vec(&header).context("failed to serialize response header")?;
    send.write_all(&(header_bytes.len() as u32).to_le_bytes()).await.context("failed to write header length")?;
    send.write_all(&header_bytes).await.context("failed to write response header")?;
    if !payload.is_empty() {
        send.write_all(&payload).await.context("failed to write response payload")?;
    }
    send.finish().context("failed to finish send stream")?;

    Ok(())
}
