use super::{ExecResult, JobExecutor};
use anyhow::{Context, Result};
use bioproof_core::ComputeJob;
use std::path::Path;
use std::pin::Pin;
use std::future::Future;
use tokio::process::Command;

// ── Nextflow executor ─────────────────────────────────────────────────────────

pub struct NextflowExecutor;

impl JobExecutor for NextflowExecutor {
    fn name(&self) -> &str { "nextflow" }

    fn probe(&self) -> bool {
        std::process::Command::new("nextflow")
            .arg("-version")
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
        Box::pin(run_nextflow(job, pipeline, input_dir, output_dir))
    }
}

// ── Snakemake executor ────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct SnakemakeExecutor;

impl JobExecutor for SnakemakeExecutor {
    fn name(&self) -> &str { "snakemake" }

    fn probe(&self) -> bool {
        std::process::Command::new("snakemake")
            .arg("--version")
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
        Box::pin(run_snakemake(job, pipeline, input_dir, output_dir))
    }
}

// ── Nextflow implementation ───────────────────────────────────────────────────

/// Execute a `.nf` pipeline with Nextflow.
///
/// Passes `--input_dir` and `--output_dir` as named parameters so pipelines
/// can reference them via `params.input_dir` / `params.output_dir`.
async fn run_nextflow(
    _job:       &ComputeJob,
    pipeline:   &Path,
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<ExecResult> {
    let out = Command::new("nextflow")
        .args([
            "run",
            pipeline.to_str().unwrap_or(""),
            "--input_dir",  input_dir.to_str().unwrap_or(""),
            "--output_dir", output_dir.to_str().unwrap_or(""),
            "-ansi-log", "false",
        ])
        .env("NXF_WORK", output_dir.join(".nextflow_work").to_str().unwrap_or("/tmp/nxf"))
        .kill_on_drop(true)
        .output()
        .await
        .context("nextflow run failed")?;

    let mut trace = String::new();
    trace.push_str(&String::from_utf8_lossy(&out.stdout));
    trace.push_str(&String::from_utf8_lossy(&out.stderr));

    Ok(ExecResult { trace, success: out.status.success() })
}

// ── Snakemake implementation ──────────────────────────────────────────────────

/// Execute a `Snakefile` with Snakemake.
///
/// Passes `input_dir` and `output_dir` as config values so the Snakefile
/// can reference them via `config["input_dir"]` / `config["output_dir"]`.
#[allow(dead_code)]
async fn run_snakemake(
    _job:       &ComputeJob,
    pipeline:   &Path,
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<ExecResult> {
    let out = Command::new("snakemake")
        .args([
            "--snakefile", pipeline.to_str().unwrap_or(""),
            "--cores",     "all",
            "--config",
            &format!("input_dir={}", input_dir.display()),
            &format!("output_dir={}", output_dir.display()),
            "--nolock",
        ])
        .kill_on_drop(true)
        .output()
        .await
        .context("snakemake run failed")?;

    let mut trace = String::new();
    trace.push_str(&String::from_utf8_lossy(&out.stdout));
    trace.push_str(&String::from_utf8_lossy(&out.stderr));

    Ok(ExecResult { trace, success: out.status.success() })
}
