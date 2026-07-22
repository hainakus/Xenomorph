//! Multi-GPU training CLI arguments for `xenom-miner`.

use clap::Parser;

/// GPU-specific training options.
#[derive(Parser, Debug, Clone)]
pub struct GpuArgs {
    /// GPUs to use (comma-separated, e.g., "0,1,2,3").
    #[arg(long, value_delimiter = ',', num_args = 1.., default_value = "0")]
    pub gpus: Vec<usize>,

    /// Micro-batch size per GPU per accumulation step.
    #[arg(long, default_value_t = 4)]
    pub micro_batch_size: usize,

    /// Gradient accumulation steps. Effective batch = micro_batch_size * gpus * accumulation.
    #[arg(long, default_value_t = 1)]
    pub gradient_accumulation: usize,

    /// Enable FP16 mixed precision on CUDA/Metal backends.
    #[arg(long)]
    pub fp16: bool,

    /// Enable gradient checkpointing (stub; currently a no-op placeholder).
    #[arg(long)]
    pub gradient_checkpointing: bool,

    /// ZeRO optimization level (0=disabled, 1=optimizer states, 2=gradients, 3=parameters).
    /// Currently only 0 is implemented; higher values are reserved.
    #[arg(long, default_value_t = 0)]
    pub zero: u8,

    /// Gradient compression ratio for FedAvg submissions. 1.0 = dense gradients;
    /// 0.1 keeps only the largest 10% of values by absolute magnitude. Lower
    /// values drastically reduce upload size over slow remote links.
    #[arg(long, default_value_t = 1.0)]
    pub gradient_top_k_ratio: f32,
}

impl GpuArgs {
    /// Validate that at least one GPU is requested and ZeRO level is supported.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.gpus.is_empty() {
            anyhow::bail!("--gpus must contain at least one device index");
        }
        if self.micro_batch_size == 0 {
            anyhow::bail!("--micro-batch-size must be > 0");
        }
        if self.gradient_accumulation == 0 {
            anyhow::bail!("--gradient-accumulation must be > 0");
        }
        if self.zero > 0 {
            anyhow::bail!("--zero > 0 is not implemented yet");
        }
        if self.gradient_top_k_ratio.is_nan() || self.gradient_top_k_ratio < 0.0 || self.gradient_top_k_ratio > 1.0 {
            anyhow::bail!("--gradient-top-k-ratio must be between 0.0 and 1.0");
        }
        Ok(())
    }
}
