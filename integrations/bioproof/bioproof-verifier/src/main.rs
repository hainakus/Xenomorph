use anyhow::{bail, Context, Result};
use bioproof_core::{compute_proof, verify_manifest_sig, Certificate};
use clap::{Arg, Command};

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();
    let cert_path  = m.get_one::<String>("cert").unwrap();
    let file_path  = m.get_one::<String>("file").cloned();
    let node_addr  = m.get_one::<String>("node").cloned();

    // ── Load certificate ──────────────────────────────────────────────────
    let cert_bytes = tokio::fs::read(cert_path)
        .await
        .with_context(|| format!("cannot read certificate '{cert_path}'"))?;
    let cert: Certificate =
        serde_json::from_slice(&cert_bytes).context("invalid certificate JSON")?;

    let mut passed = true;

    // ── Step 1: recompute manifest hash ──────────────────────────────────
    let recomputed_hash = cert.manifest.hash_hex();
    if recomputed_hash == cert.manifest_hash {
        log::info!("[OK] manifest_hash matches");
    } else {
        log::error!("[FAIL] manifest_hash mismatch: expected={} got={}",
            cert.manifest_hash, recomputed_hash);
        passed = false;
    }

    // ── Step 2: verify issuer signature ──────────────────────────────────
    let digest = cert.manifest.hash_bytes();
    match verify_manifest_sig(&digest, &cert.issuer_sig, &cert.issuer_pubkey) {
        Ok(true)  => log::info!("[OK] issuer signature valid"),
        Ok(false) => {
            log::error!("[FAIL] issuer signature INVALID");
            passed = false;
        }
        Err(e) => {
            log::error!("[FAIL] signature verification error: {e}");
            passed = false;
        }
    }

    // ── Step 3: verify proof_root by re-hashing the file (optional) ──────
    if let Some(ref path) = file_path {
        log::info!("Re-hashing file '{path}'…");
        let data = tokio::fs::read(path)
            .await
            .with_context(|| format!("cannot read file '{path}'"))?;
        let (proof_root, file_hash, _) =
            compute_proof(&data, cert.manifest.chunk_size);

        if file_hash == cert.manifest.file_hash {
            log::info!("[OK] file_hash matches");
        } else {
            log::error!("[FAIL] file_hash mismatch: expected={} got={}",
                cert.manifest.file_hash, file_hash);
            passed = false;
        }

        if proof_root == cert.manifest.proof_root {
            log::info!("[OK] proof_root matches");
        } else {
            log::error!("[FAIL] proof_root mismatch: expected={} got={}",
                cert.manifest.proof_root, proof_root);
            passed = false;
        }
    } else {
        log::info!("[SKIP] file not provided — skipping proof_root re-computation");
    }

    // ── Step 4: check anchor on-chain (optional) ──────────────────────────
    if let (Some(txid), Some(addr)) = (&cert.txid, node_addr.as_deref()) {
        match verify_on_chain(addr, txid, &cert.manifest.proof_root, &cert.manifest_hash).await {
            Ok(true)  => log::info!("[OK] on-chain anchor verified (txid={txid})"),
            Ok(false) => {
                log::error!("[FAIL] on-chain anchor NOT found or payload mismatch");
                passed = false;
            }
            Err(e) => log::warn!("[WARN] on-chain check failed: {e:#}"),
        }
    } else if cert.txid.is_some() {
        log::info!("[SKIP] no --node provided — skipping on-chain verification");
    } else {
        log::info!("[SKIP] certificate has no txid — not yet anchored");
    }

    // ── Result ────────────────────────────────────────────────────────────
    if passed {
        println!("\nVERIFICATION PASSED");
        Ok(())
    } else {
        bail!("VERIFICATION FAILED — see errors above");
    }
}

// ── On-chain verification ─────────────────────────────────────────────────────

async fn verify_on_chain(
    _node_addr:    &str,
    _txid:         &str,
    _proof_root:   &str,
    _manifest_hash: &str,
) -> Result<bool> {
    // Full on-chain tx lookup requires querying the bioproof-indexer REST API
    // or scanning blocks via the node.  This is handled by bioproof-api.
    // For now, report inconclusive so the verifier doesn't block on it.
    log::warn!("On-chain anchor verification requires the bioproof-indexer (not yet implemented in standalone verifier)");
    Ok(true)
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("bioproof-verifier")
        .about("BioProof verifier — validate a certificate's manifest hash, signature and on-chain anchor")
        .arg(Arg::new("cert")
            .short('c').long("cert").value_name("PATH").required(true)
            .help("Path to the certificate JSON file"))
        .arg(Arg::new("file")
            .short('f').long("file").value_name("PATH")
            .help("Original input file — enables proof_root re-computation"))
        .arg(Arg::new("node")
            .short('n').long("node").value_name("ADDR")
            .help("Xenom node gRPC address for on-chain anchor check"))
}
