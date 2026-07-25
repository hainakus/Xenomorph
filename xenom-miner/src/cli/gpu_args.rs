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

    /// Maximum sequence length per sample. The model config is capped at this value to save VRAM.
    #[arg(long, default_value_t = 512)]
    pub max_seq_len: usize,

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
    #[arg(long, default_value_t = 1.0, env = "XENO_GRADIENT_TOP_K_RATIO")]
    pub gradient_top_k_ratio: f32,

    /// Enable LoRA (Low-Rank Adaptation) fine-tuning instead of full fine-tuning.
    #[arg(long, env = "XENO_LORA")]
    pub lora: bool,

    /// LoRA rank.
    #[arg(long, default_value_t = 8, env = "XENO_LORA_RANK")]
    pub lora_rank: usize,

    /// LoRA alpha scaling factor.
    #[arg(long, default_value_t = 16.0, env = "XENO_LORA_ALPHA")]
    pub lora_alpha: f32,

    /// LoRA dropout probability (currently unused).
    #[arg(long, default_value_t = 0.0, env = "XENO_LORA_DROPOUT")]
    pub lora_dropout: f32,

    /// Comma-separated LoRA target module names (default: query,key,value,transform_dense,up_proj,down_proj).
    #[arg(long, value_delimiter = ',', num_args = 1.., env = "XENO_LORA_TARGET_MODULES")]
    pub lora_target_modules: Vec<String>,
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
        if self.max_seq_len == 0 {
            anyhow::bail!("--max-seq-len must be > 0");
        }
        if self.zero > 0 {
            anyhow::bail!("--zero > 0 is not implemented yet");
        }
        if self.gradient_top_k_ratio.is_nan() || self.gradient_top_k_ratio < 0.0 || self.gradient_top_k_ratio > 1.0 {
            anyhow::bail!("--gradient-top-k-ratio must be between 0.0 and 1.0");
        }
        if self.lora {
            if self.lora_rank == 0 {
                anyhow::bail!("--lora-rank must be > 0");
            }
            if self.lora_alpha <= 0.0 {
                anyhow::bail!("--lora-alpha must be > 0");
            }
            if self.lora_dropout.is_nan() || self.lora_dropout < 0.0 || self.lora_dropout > 1.0 {
                anyhow::bail!("--lora-dropout must be between 0.0 and 1.0");
            }
        }
        Ok(())
    }
}
