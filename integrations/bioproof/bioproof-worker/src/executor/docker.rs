use super::{ExecResult, JobExecutor};
use anyhow::{Context, Result};
use bioproof_core::ComputeJob;
use std::path::Path;
use std::pin::Pin;
use std::future::Future;
use tokio::process::Command;

// ── Docker executor ───────────────────────────────────────────────────────────

pub struct DockerExecutor;

impl JobExecutor for DockerExecutor {
    fn name(&self) -> &str { "docker" }

    fn probe(&self) -> bool {
        std::process::Command::new("docker")
            .arg("info")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn execute<'a>(
        &'a self,
        job:        &'a ComputeJob,
        pipeline:   &'a Path,
        input_dir:  &'a Path,
        output_dir: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<ExecResult>> + Send + 'a>> {
        Box::pin(run_docker(job, pipeline, input_dir, output_dir, false))
    }
}

// ── Singularity / Apptainer executor ─────────────────────────────────────────

pub struct SingularityExecutor;

impl JobExecutor for SingularityExecutor {
    fn name(&self) -> &str { "singularity" }

    fn probe(&self) -> bool {
        std::process::Command::new("singularity")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or_else(|_| {
                // Also accept apptainer (the rebranded version)
                std::process::Command::new("apptainer")
                    .arg("--version")
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
            })
    }

    fn execute<'a>(
        &'a self,
        job:        &'a ComputeJob,
        pipeline:   &'a Path,
        input_dir:  &'a Path,
        output_dir: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<ExecResult>> + Send + 'a>> {
        Box::pin(run_docker(job, pipeline, input_dir, output_dir, true))
    }
}

// ── Shared implementation ─────────────────────────────────────────────────────

/// Run the pipeline inside a container.
///
/// The pipeline file is expected to be a shell script that receives:
///   INPUT_DIR  — bind-mounted at /data/input
///   OUTPUT_DIR — bind-mounted at /data/output
///
/// The container image is determined from `job.container_hash` (treated as an
/// image reference like `docker.io/biocontainers/gatk4:4.5.0.0`).
/// If absent, the pipeline script is run in the bioproof/runner base image.
async fn run_docker(
    job:        &ComputeJob,
    pipeline:   &Path,
    input_dir:  &Path,
    output_dir: &Path,
    singularity: bool,
) -> Result<ExecResult> {
    let image = job.container_hash.as_deref().unwrap_or("bioproof/runner:latest");

    let input_abs  = input_dir.canonicalize()
        .with_context(|| format!("cannot resolve {}", input_dir.display()))?;
    let output_abs = output_dir.canonicalize()
        .with_context(|| format!("cannot resolve {}", output_dir.display()))?;
    let pipeline_abs = pipeline.canonicalize()
        .with_context(|| format!("cannot resolve {}", pipeline.display()))?;

    let out = if singularity {
        let bin = if std::path::Path::new("/usr/bin/apptainer").exists() {
            "apptainer"
        } else {
            "singularity"
        };
        Command::new(bin)
            .args([
                "exec",
                "--bind", &format!("{}:/data/input:ro",  input_abs.display()),
                "--bind", &format!("{}:/data/output",    output_abs.display()),
                "--bind", &format!("{}:/pipeline/run.sh:ro", pipeline_abs.display()),
                "--env",  "INPUT_DIR=/data/input",
                "--env",  "OUTPUT_DIR=/data/output",
                image,
                "bash", "/pipeline/run.sh",
            ])
            .output()
            .await
            .context("singularity exec failed")?
    } else {
        Command::new("docker")
            .args([
                "run", "--rm",
                "-v", &format!("{}:/data/input:ro",  input_abs.display()),
                "-v", &format!("{}:/data/output",    output_abs.display()),
                "-v", &format!("{}:/pipeline/run.sh:ro", pipeline_abs.display()),
                "-e", "INPUT_DIR=/data/input",
                "-e", "OUTPUT_DIR=/data/output",
                image,
                "bash", "/pipeline/run.sh",
            ])
            .output()
            .await
            .context("docker run failed")?
    };

    let mut trace = String::new();
    trace.push_str(&String::from_utf8_lossy(&out.stdout));
    trace.push_str(&String::from_utf8_lossy(&out.stderr));

    Ok(ExecResult { trace, success: out.status.success() })
}
