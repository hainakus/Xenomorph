use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use borsh::{to_vec, BorshDeserialize};
use quinn::{ClientConfig, Endpoint};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as RustlsError, SignatureScheme};

use crate::protocol::{CheckpointFileRequest, CheckpointFileResponseHeader, ResponseStatus};

/// QUIC client that downloads checkpoint files from a peer.
pub struct CheckpointTransferClient {
    endpoint: Endpoint,
    server_name: String,
}

impl CheckpointTransferClient {
    /// Create a client endpoint bound to `local_addr`.
    ///
    /// `server_name` must match the DNS name in the server certificate. For the
    /// self-signed certificates used in this crate, `"localhost"` is typical.
    pub fn new(local_addr: SocketAddr, server_name: &str) -> Result<Self> {
        let client_config = configure_client()?;
        let mut endpoint = Endpoint::client(local_addr).context("failed to bind QUIC client endpoint")?;
        endpoint.set_default_client_config(client_config);
        Ok(Self { endpoint, server_name: server_name.to_string() })
    }

    /// Request a single checkpoint file from `server_addr`.
    pub async fn get_file(&self, server_addr: SocketAddr, request: &CheckpointFileRequest) -> Result<Vec<u8>> {
        let conn = self
            .endpoint
            .connect(server_addr, &self.server_name)
            .context("failed to initiate QUIC connection")?
            .await
            .context("QUIC connection failed")?;
        let (mut send, mut recv) = conn.open_bi().await.context("failed to open QUIC bidirectional stream")?;

        let req_bytes = to_vec(request).context("failed to serialize request")?;
        send.write_all(&(req_bytes.len() as u32).to_le_bytes()).await.context("failed to write request length")?;
        send.write_all(&req_bytes).await.context("failed to write request body")?;
        send.finish().context("failed to finish send stream")?;

        let mut header_len_bytes = [0u8; 4];
        recv.read_exact(&mut header_len_bytes).await.context("failed to read response header length")?;
        let header_len = u32::from_le_bytes(header_len_bytes) as usize;

        let mut header_buf = vec![0u8; header_len];
        recv.read_exact(&mut header_buf).await.context("failed to read response header")?;
        let header = CheckpointFileResponseHeader::try_from_slice(&header_buf).context("failed to deserialize response header")?;

        match header.status {
            ResponseStatus::Ok => {
                let mut payload = vec![0u8; header.length as usize];
                if header.length > 0 {
                    recv.read_exact(&mut payload).await.context("failed to read response payload")?;
                }
                Ok(payload)
            }
            ResponseStatus::NotFound => bail!("checkpoint file not found"),
            ResponseStatus::NotAuthorized => bail!("checkpoint file not authorized"),
        }
    }
}

#[derive(Debug)]
struct SkipServerVerification;

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, RustlsError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::ED25519,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
        ]
    }
}

fn configure_client() -> Result<ClientConfig> {
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
    let quic_config =
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto).context("failed to create QUIC client crypto config")?;
    Ok(ClientConfig::new(Arc::new(quic_config)))
}
