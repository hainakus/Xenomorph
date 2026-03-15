use anyhow::{bail, Context, Result};
use kaspa_addresses::Prefix;
use bioproof_core::{
    compute_proof, sign_manifest, AnchorPayload, ArtifactType, BioProofKeypair, Certificate,
    Manifest,
};
use clap::{Arg, Command};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();

    // ── Inputs ──────────────────────────────────────────────────────────────
    let file_path    = m.get_one::<String>("file").unwrap();
    let dataset_id   = m.get_one::<String>("dataset-id").unwrap().clone();
    let artifact_str = m.get_one::<String>("artifact-type").unwrap();
    let issuer       = m.get_one::<String>("issuer").unwrap().clone();
    let privkey_hex  = m.get_one::<String>("private-key").unwrap();
    let chunk_size: usize = m
        .get_one::<String>("chunk-size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(bioproof_core::types::DEFAULT_CHUNK_SIZE);
    let parent_root  = m.get_one::<String>("parent-root").cloned();
    let pipeline_hash = m.get_one::<String>("pipeline-hash").cloned();
    let model_hash   = m.get_one::<String>("model-hash").cloned();
    let node_address = m.get_one::<String>("node").cloned();
    let dry_run      = !m.get_flag("submit");
    let out_path     = m.get_one::<String>("out").cloned();
    let prefix = if m.get_flag("devnet") {
        Prefix::Devnet
    } else if m.get_flag("testnet") {
        Prefix::Testnet
    } else {
        Prefix::Mainnet
    };

    let artifact_type = ArtifactType::from_str(artifact_str).unwrap();

    // ── Load keypair ────────────────────────────────────────────────────────
    let keypair = BioProofKeypair::from_hex(privkey_hex)
        .context("invalid --private-key (expected 64 hex chars)")?;

    // ── Read file ───────────────────────────────────────────────────────────
    log::info!("Reading {file_path}…");
    let data = tokio::fs::read(file_path)
        .await
        .with_context(|| format!("cannot read file '{file_path}'"))?;
    log::info!("  {} bytes", data.len());

    // ── Compute proof ───────────────────────────────────────────────────────
    log::info!("Computing BLAKE3 proof (chunk_size={chunk_size})…");
    let (proof_root, file_hash, _chunks) = compute_proof(&data, chunk_size);
    log::info!("  proof_root   = {proof_root}");
    log::info!("  file_hash    = {file_hash}");

    // ── Build manifest ──────────────────────────────────────────────────────
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let manifest = Manifest {
        dataset_id,
        artifact_type,
        chunk_size,
        file_hash,
        proof_root: proof_root.clone(),
        pipeline_hash,
        model_hash,
        parent_root,
        issuer,
        created_at,
    };

    // ── Sign manifest ────────────────────────────────────────────────────────
    let manifest_hash = manifest.hash_hex();
    let digest        = manifest.hash_bytes();
    let issuer_sig    = sign_manifest(&digest, privkey_hex)
        .context("signing failed")?;
    let issuer_pubkey = keypair.pubkey_hex();

    log::info!("  manifest_hash = {manifest_hash}");
    log::info!("  issuer_pubkey = {issuer_pubkey}");

    // ── Build AnchorPayload (on-chain OP_RETURN content) ─────────────────────
    let payload = AnchorPayload::new(&proof_root, &manifest_hash, &manifest.artifact_type.to_string());
    let op_return_bytes = payload.to_op_return_bytes();
    log::info!("  OP_RETURN payload: {} bytes", op_return_bytes.len());

    // ── Build certificate ────────────────────────────────────────────────────
    let mut cert = Certificate {
        manifest,
        manifest_hash,
        issuer_sig,
        issuer_pubkey,
        txid:        None,
        daa_score:   None,
        anchored_at: None,
    };

    // ── Submit to chain ──────────────────────────────────────────────────────
    if dry_run {
        log::info!("Dry-run mode (pass --submit --node <addr> to anchor on-chain)");
    } else {
        let node_addr = node_address
            .as_deref()
            .unwrap_or("grpc://localhost:36669");

        log::info!("Submitting anchor to {node_addr}…");
        let txid = submit_anchor(node_addr, &op_return_bytes, privkey_hex, prefix).await?;
        log::info!("  txid = {txid}");
        cert.txid = Some(txid);
    }

    // ── Output certificate ───────────────────────────────────────────────────
    let cert_json = serde_json::to_string_pretty(&cert)?;
    if let Some(ref path) = out_path {
        tokio::fs::write(path, &cert_json)
            .await
            .with_context(|| format!("cannot write certificate to '{path}'"))?;
        log::info!("Certificate written to {path}");
    } else {
        println!("{cert_json}");
    }

    Ok(())
}

// ── Transaction submission ────────────────────────────────────────────────────

async fn submit_anchor(
    node_addr:       &str,
    op_return_bytes: &[u8],
    privkey_hex:     &str,
    prefix:          Prefix,
) -> Result<String> {
    use kaspa_grpc_client::GrpcClient;
    use std::sync::Arc;

    let url = if node_addr.starts_with("grpc://") {
        node_addr.to_owned()
    } else {
        format!("grpc://{node_addr}")
    };

    let rpc     = Arc::new(GrpcClient::connect(url).await.context("cannot connect to Xenom node")?);
    let keypair = xenom_anchor_client::keypair_from_hex(privkey_hex)?;

    log::info!(
        "Funding address: {}",
        xenom_anchor_client::address_from_keypair(&keypair, prefix)
    );

    xenom_anchor_client::submit_anchor(
        &rpc,
        &keypair,
        op_return_bytes,
        xenom_anchor_client::DEFAULT_FEE_PER_INPUT,
        prefix,
    ).await
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("bioproof-daemon")
        .about("BioProof anchor daemon — chunk, sign and anchor genomics/AI artefacts on Xenom")
        .arg(Arg::new("file")
            .short('f').long("file").value_name("PATH").required(true)
            .help("Input file (FASTQ, BAM, VCF, pipeline, model weights, …)"))
        .arg(Arg::new("dataset-id")
            .short('d').long("dataset-id").value_name("ID").required(true)
            .help("Stable identifier for this dataset (e.g. sample barcode)"))
        .arg(Arg::new("artifact-type")
            .short('t').long("artifact-type").value_name("TYPE").required(true)
            .help("Artefact type: fastq|bam|cram|vcf|pipeline|ai-model|ai-output|report|<custom>"))
        .arg(Arg::new("issuer")
            .short('i').long("issuer").value_name("ID").required(true)
            .help("Issuer identifier (lab ID, DID, public key fingerprint)"))
        .arg(Arg::new("private-key")
            .short('k').long("private-key").value_name("HEX").required(true)
            .help("secp256k1 private key as 64-char hex"))
        .arg(Arg::new("chunk-size")
            .long("chunk-size").value_name("BYTES")
            .default_value("4194304")
            .help("Chunk size for BLAKE3 Merkle tree (bytes, default 4 MiB)"))
        .arg(Arg::new("parent-root")
            .long("parent-root").value_name("HEX")
            .help("proof_root of the parent artefact (for lineage)"))
        .arg(Arg::new("pipeline-hash")
            .long("pipeline-hash").value_name("HEX")
            .help("BLAKE3 hash of the pipeline definition file"))
        .arg(Arg::new("model-hash")
            .long("model-hash").value_name("HEX")
            .help("BLAKE3 hash of the AI model weights"))
        .arg(Arg::new("node")
            .short('n').long("node").value_name("ADDR")
            .default_value("grpc://localhost:36669")
            .help("Xenom node gRPC address"))
        .arg(Arg::new("submit")
            .long("submit")
            .action(clap::ArgAction::SetTrue)
            .help("Actually submit the anchor transaction (default: dry-run)"))
        .arg(Arg::new("out")
            .short('o').long("out").value_name("PATH")
            .help("Write certificate JSON to file instead of stdout"))
        .arg(Arg::new("devnet")
            .long("devnet")
            .action(clap::ArgAction::SetTrue)
            .help("Use devnet address prefix (xenomdev:)"))
        .arg(Arg::new("testnet")
            .long("testnet")
            .action(clap::ArgAction::SetTrue)
            .help("Use testnet address prefix (xenomtest:)"))
}
