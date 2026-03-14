use super::{ExecResult, JobExecutor};
use anyhow::{Context, Result};
use bioproof_core::ComputeJob;
use std::path::Path;
use std::pin::Pin;
use std::future::Future;
use tokio::process::Command;

/// AI executor: runs PyTorch / TensorFlow / CUDA scripts.
///
/// Probes for `python3` with torch available (GPU optional but preferred).
/// Sets `CUDA_VISIBLE_DEVICES` to expose all GPUs or restricts to a subset.
pub struct AiExecutor;

impl JobExecutor for AiExecutor {
    fn name(&self) -> &str { "ai" }

    fn probe(&self) -> bool {
        // Check python3 + torch importable
        std::process::Command::new("python3")
            .args(["-c", "import torch"])
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
        Box::pin(run_ai(job, pipeline, input_dir, output_dir))
    }
}

async fn run_ai(
    job:        &ComputeJob,
    pipeline:   &Path,
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<ExecResult> {
    let ext = pipeline.extension().and_then(|e| e.to_str()).unwrap_or("py");

    let mut cmd = match ext {
        "py" => {
            let mut c = Command::new("python3");
            c.arg(pipeline);
            c
        }
        "sh" => {
            let mut c = Command::new("bash");
            c.arg(pipeline);
            c
        }
        _ => {
            let mut c = Command::new("python3");
            c.arg(pipeline);
            c
        }
    };

    // Pass job spec as environment so the script can read model / config paths
    cmd.env("INPUT_DIR",      input_dir)
       .env("OUTPUT_DIR",     output_dir)
       .env("JOB_ID",         &job.job_id)
       .env("JOB_TYPE",       job.job_type.to_string())
       .env("CUDA_VISIBLE_DEVICES", "all")  // expose all GPUs
       .kill_on_drop(true);

    // Pass model hash so the script can verify / download the right weights
    if let Some(ref mh) = job.model_hash {
        cmd.env("MODEL_HASH", mh);
    }

    let out = cmd.output().await.context("AI pipeline failed to spawn")?;

    let mut trace = String::new();
    trace.push_str(&String::from_utf8_lossy(&out.stdout));
    trace.push_str(&String::from_utf8_lossy(&out.stderr));

    Ok(ExecResult { trace, success: out.status.success() })
}
