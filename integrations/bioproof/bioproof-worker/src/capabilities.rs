use anyhow::Result;
use bioproof_core::{ExecutionBackend, GpuInfo, JobType, WorkerCapabilities};
use std::time::{SystemTime, UNIX_EPOCH};
use sysinfo::System;
use tokio::process::Command;

/// Detect all hardware and installed software on this machine.
pub async fn detect(worker_pubkey: String, max_concurrency: usize) -> Result<WorkerCapabilities> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let hostname = System::host_name().unwrap_or_else(|| "unknown".to_owned());
    let cpu_count = sys.cpus().len();
    let ram_mib = sys.total_memory() / 1024 / 1024;

    let disk_mib = available_disk_mib().await;
    let gpus = detect_gpus().await;
    let backends = detect_backends().await;
    let job_types = infer_job_types(&backends, &gpus);

    let measured_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(WorkerCapabilities {
        worker_pubkey,
        hostname,
        cpu_count,
        ram_mib,
        disk_mib,
        gpus,
        backends,
        job_types,
        max_concurrency,
        measured_at,
    })
}

// ── GPU detection ─────────────────────────────────────────────────────────────

async fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // Try nvidia-smi (CUDA)
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .await
    {
        if out.status.success() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                if parts.len() >= 3 {
                    let index   = parts[0].parse().unwrap_or(0);
                    let model   = parts[1].to_owned();
                    let vram_mb = parts[2].parse().unwrap_or(0);
                    gpus.push(GpuInfo { index, model, vram_mb, cuda: true, rocm: false });
                }
            }
        }
    }

    // Try rocm-smi (AMD ROCm) if no NVIDIA found
    if gpus.is_empty() {
        if let Ok(out) = Command::new("rocm-smi").args(["--showproductname"]).output().await {
            if out.status.success() {
                for (i, line) in String::from_utf8_lossy(&out.stdout).lines().enumerate() {
                    if line.contains("GPU") || line.contains("Radeon") || line.contains("gfx") {
                        gpus.push(GpuInfo {
                            index:   i,
                            model:   line.trim().to_owned(),
                            vram_mb: 0,
                            cuda:    false,
                            rocm:    true,
                        });
                    }
                }
            }
        }
    }

    gpus
}

// ── Backend detection ─────────────────────────────────────────────────────────

async fn detect_backends() -> Vec<ExecutionBackend> {
    let probes: &[(&str, &[&str], ExecutionBackend)] = &[
        ("docker",      &["--version"],     ExecutionBackend::Docker),
        ("singularity", &["--version"],     ExecutionBackend::Singularity),
        ("apptainer",   &["--version"],     ExecutionBackend::Singularity),
        ("nextflow",    &["-version"],      ExecutionBackend::Nextflow),
        ("snakemake",   &["--version"],     ExecutionBackend::Snakemake),
        ("cromwell",    &["--version"],     ExecutionBackend::Cromwell),
        ("python3",     &["--version"],     ExecutionBackend::Native),
        ("bash",        &["--version"],     ExecutionBackend::Native),
    ];

    let mut found = Vec::new();
    for (bin, args, backend) in probes {
        if Command::new(bin).args(*args).output().await
            .map(|o| o.status.success())
            .unwrap_or(false)
            && !found.contains(backend)
        {
            found.push(backend.clone());
        }
    }

    // Check for CUDA toolkit
    if Command::new("nvcc").arg("--version").output().await
        .map(|o| o.status.success()).unwrap_or(false)
    {
        found.push(ExecutionBackend::Cuda);
    }
    // Check for ROCm HIP
    if Command::new("hipcc").arg("--version").output().await
        .map(|o| o.status.success()).unwrap_or(false)
    {
        found.push(ExecutionBackend::Rocm);
    }

    found
}

// ── Job type inference ────────────────────────────────────────────────────────

fn infer_job_types(backends: &[ExecutionBackend], gpus: &[GpuInfo]) -> Vec<JobType> {
    let mut types = Vec::new();
    let has_container = backends.contains(&ExecutionBackend::Docker)
        || backends.contains(&ExecutionBackend::Singularity);
    let has_workflow  = backends.contains(&ExecutionBackend::Nextflow)
        || backends.contains(&ExecutionBackend::Snakemake)
        || backends.contains(&ExecutionBackend::Cromwell);
    let has_native    = backends.contains(&ExecutionBackend::Native);
    let has_gpu       = !gpus.is_empty();
    let has_cuda      = gpus.iter().any(|g| g.cuda)
        || backends.contains(&ExecutionBackend::Cuda);
    let has_rocm      = gpus.iter().any(|g| g.rocm)
        || backends.contains(&ExecutionBackend::Rocm);

    if has_container || has_workflow || has_native {
        types.push(JobType::GenomicsPipeline);
        types.push(JobType::VariantCalling);
        types.push(JobType::GenomeAssembly);
        types.push(JobType::SingleCellRna);
        types.push(JobType::Metagenomics);
    }

    if has_gpu {
        types.push(JobType::AiInference);
        types.push(JobType::ProteinFolding);
    }

    if has_cuda || has_rocm {
        types.push(JobType::AiTraining);
        types.push(JobType::MolecularDynamics);
    }

    types.dedup();
    types
}

// ── Disk space ───────────────────────────────────────────────────────────────

async fn available_disk_mib() -> u64 {
    if let Ok(out) = Command::new("df")
        .args(["-m", "--output=avail", "."])
        .output()
        .await
    {
        if out.status.success() {
            let txt = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = txt.lines().nth(1) {
                return line.trim().parse().unwrap_or(0);
            }
        }
    }
    0
}
