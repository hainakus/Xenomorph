pub mod ai;
pub mod docker;
pub mod native;
pub mod nextflow;

use anyhow::Result;
use bioproof_core::{ComputeJob, ExecutionBackend};
use std::path::Path;

// ── JobExecutor trait ─────────────────────────────────────────────────────────

/// Result of running a job execution backend.
pub struct ExecResult {
    /// Combined stdout + stderr captured during execution.
    pub trace:   String,
    /// Whether the pipeline exited successfully.
    pub success: bool,
}

/// Trait implemented by each execution backend.
pub trait JobExecutor: Send + Sync {
    /// Backend identifier (for logging).
    fn name(&self) -> &str;

    /// Check that the required tooling is actually present on this machine.
    fn probe(&self) -> bool;

    /// Execute the job pipeline synchronously.
    ///
    /// - `job`        — the job spec
    /// - `pipeline`   — path to the pipeline entry-point file
    /// - `input_dir`  — directory containing input data
    /// - `output_dir` — directory where outputs must be written
    fn execute<'a>(
        &'a self,
        job:        &'a ComputeJob,
        pipeline:   &'a Path,
        input_dir:  &'a Path,
        output_dir: &'a Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExecResult>> + Send + 'a>>;
}

// ── Dispatch: pick the best executor for a job ────────────────────────────────

/// Select and return the appropriate executor given the preferred backend and
/// what is actually installed on the machine.
///
/// Priority: explicit preference → Nextflow → Docker → Singularity → Native.
pub fn pick_executor(preferred: Option<&ExecutionBackend>) -> Box<dyn JobExecutor> {
    let order: &[ExecutionBackend] = preferred
        .map(std::slice::from_ref)
        .unwrap_or(&[]);

    let candidates: Vec<Box<dyn JobExecutor>> = vec![
        Box::new(nextflow::NextflowExecutor),
        Box::new(docker::DockerExecutor),
        Box::new(docker::SingularityExecutor),
        Box::new(ai::AiExecutor),
        Box::new(native::NativeExecutor),
    ];

    // Respect explicit preference first.
    for pref in order {
        for c in &candidates {
            if backend_matches(c.name(), pref) && c.probe() {
                log::debug!("Executor selected (preferred): {}", c.name());
            }
        }
    }

    // Fall through to first available.
    for c in candidates {
        if c.probe() {
            log::debug!("Executor selected (auto): {}", c.name());
            return c;
        }
    }

    // Always-available fallback.
    log::warn!("No specialised executor found; falling back to native");
    Box::new(native::NativeExecutor)
}

fn backend_matches(executor_name: &str, backend: &ExecutionBackend) -> bool {
    match backend {
        ExecutionBackend::Docker      => executor_name == "docker",
        ExecutionBackend::Singularity => executor_name == "singularity",
        ExecutionBackend::Nextflow    => executor_name == "nextflow",
        ExecutionBackend::Snakemake   => executor_name == "snakemake",
        ExecutionBackend::Cromwell    => executor_name == "cromwell",
        ExecutionBackend::Native      => executor_name == "native",
        ExecutionBackend::Cuda
        | ExecutionBackend::Rocm      => executor_name == "ai",
    }
}
