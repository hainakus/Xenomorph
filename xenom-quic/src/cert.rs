use anyhow::{Context, Result};
use rcgen::{generate_simple_self_signed, CertifiedKey};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// Generate a self-signed X.509 certificate and private key for use by a QUIC server.
///
/// `subject_alt_names` should include the DNS names or host identifiers clients will use
/// to connect (e.g. "localhost").
pub fn generate_self_signed_cert(subject_alt_names: &[String]) -> Result<(CertificateDer<'static>, PrivateKeyDer<'static>)> {
    let CertifiedKey { cert, key_pair } =
        generate_simple_self_signed(subject_alt_names.to_vec()).context("failed to generate self-signed certificate")?;
    let cert_der = CertificateDer::from(cert);
    let key_der = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(key_pair.serialize_der()));
    Ok((cert_der, key_der))
}
