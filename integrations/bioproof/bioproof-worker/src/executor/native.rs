use super::{ExecResult, JobExecutor};
use anyhow::{Context, Result};
use bioproof_core::ComputeJob;
use std::path::Path;
use std::pin::Pin;
use std::future::Future;
use tokio::process::Command;

/// Native executor: bash / python3 / conda — no container required.
/// Entry-point file extension determines the interpreter:
///   .sh  → bash
///   .py  → python3
///   .nf  → nextflow (fallback if NextflowExecutor is unavailable)
///   *    → bash
pub struct NativeExecutor;

impl JobExecutor for NativeExecutor {
    fn name(&self) -> &str { "native" }

    fn probe(&self) -> bool { true } // always available as last resort

    fn execute<'a>(
        &'a self,
        job:        &'a ComputeJob,
        pipeline:   &'a Path,
        input_dir:  &'a Path,
        output_dir: &'a Path,
    ) -> Pin<Box<dyn Future<Output = Result<ExecResult>> + Send + 'a>> {
        Box::pin(run_native(job, pipeline, input_dir, output_dir))
    }
}

async fn run_native(
    _job:       &ComputeJob,
    pipeline:   &Path,
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<ExecResult> {
    let ext = pipeline.extension().and_then(|e| e.to_str()).unwrap_or("sh");

    let mut cmd = match ext {
        "py" => {
            let mut c = Command::new("python3");
            c.arg(pipeline);
            c
        }
        _ => {
            let mut c = Command::new("bash");
            c.arg(pipeline);
            c
        }
    };

    cmd.env("INPUT_DIR",  input_dir)
       .env("OUTPUT_DIR", output_dir)
       .kill_on_drop(true);

    let out = cmd.output().await.context("native pipeline failed to spawn")?;

    let mut trace = String::new();
    trace.push_str(&String::from_utf8_lossy(&out.stdout));
    trace.push_str(&String::from_utf8_lossy(&out.stderr));

    Ok(ExecResult { trace, success: out.status.success() })
}
