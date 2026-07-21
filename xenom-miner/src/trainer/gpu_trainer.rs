use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use tracing::{info, warn};

use crate::gpu::monitor::GpuMonitor;
use crate::model::DnaBert2Config;
use crate::rpc::messages::{GenomeTrainingBatchMsg, TrainingBatch};
use crate::tokenizer::DnaTokenizer;
use crate::trainer::{DeviceInfo, DeviceType, DnaBert2Trainer, GradientUpdate, Trainer, TrainingResult};

/// GPU backend selector for the DNABERT-2 trainer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    /// Auto-detect CUDA, Metal, or fall back to CPU.
    Auto,
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
    /// AMD ROCm/HIP (not yet implemented).
    Rocm,
}

/// DNABERT-2 trainer that runs on a GPU (CUDA/Metal) with CPU fallback.
///
/// `GpuTrainer` wraps the device-agnostic `DnaBert2Trainer` and handles
/// device selection, FP16 mixed precision, and GPU monitoring.
pub struct GpuTrainer {
    inner: DnaBert2Trainer,
    device_type: DeviceType,
    device_name: String,
    device_index: usize,
    threads: usize,
}

impl GpuTrainer {
    /// Load a trainable DNABERT-2 model onto the selected GPU backend.
    pub fn new(
        config: DnaBert2Config,
        weights: Vec<u8>,
        tokenizer: DnaTokenizer,
        backend: GpuBackend,
        device_index: usize,
        fp16: bool,
        threads: usize,
    ) -> Result<Self> {
        let (device, device_type, device_name) = Self::select_device(backend, device_index)?;

        let dtype = if fp16 && (device.is_cuda() || device.is_metal()) {
            info!("Using FP16 mixed precision on GPU device");
            DType::F16
        } else if fp16 {
            warn!("FP16 requested but the selected device does not support it; using F32");
            DType::F32
        } else {
            DType::F32
        };

        let inner = DnaBert2Trainer::new(config, weights, tokenizer, device, threads, dtype)
            .context("Failed to initialize DNABERT-2 trainer on selected device")?;

        Ok(Self { inner, device_type, device_name, device_index, threads })
    }

    fn select_device(backend: GpuBackend, _index: usize) -> Result<(Device, DeviceType, String)> {
        match backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                match Device::new_cuda(_index) {
                    Ok(device) => return Ok((device, DeviceType::Cuda, format!("NVIDIA CUDA device {}", _index))),
                    Err(e) => bail!("CUDA device {} is not accessible: {}. Check NVIDIA drivers/runtime.", _index, e),
                }
                #[cfg(not(feature = "cuda"))]
                bail!("CUDA support was not compiled into this binary. Rebuild with: cargo build --release -p xenom-miner --features cuda")
            }
            GpuBackend::Metal => {
                #[cfg(feature = "metal")]
                match Device::new_metal(_index) {
                    Ok(device) => return Ok((device, DeviceType::Metal, format!("Apple Metal device {}", _index))),
                    Err(e) => bail!("Metal device {} is not accessible: {}. Check macOS Metal support.", _index, e),
                }
                #[cfg(not(feature = "metal"))]
                bail!("Metal support was not compiled into this binary. Rebuild with: cargo build --release -p xenom-miner --features metal")
            }
            GpuBackend::Rocm => {
                bail!("ROCm/HIP backend is not yet supported. Use --trainer cuda or --trainer cpu")
            }
            GpuBackend::Auto => {
                #[cfg(feature = "cuda")]
                if let Ok(device) = Device::new_cuda(_index) {
                    info!("Auto-selected NVIDIA CUDA device {}", _index);
                    return Ok((device, DeviceType::Cuda, format!("NVIDIA CUDA device {}", _index)));
                }
                #[cfg(feature = "metal")]
                if let Ok(device) = Device::new_metal(_index) {
                    info!("Auto-selected Apple Metal device {}", _index);
                    return Ok((device, DeviceType::Metal, format!("Apple Metal device {}", _index)));
                }

                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                warn!(
                    "No GPU backend was compiled into this binary. Rebuild with one of:\n\
                     cargo build --release -p xenom-miner --features cuda\n\
                     cargo build --release -p xenom-miner --features metal\n\
                     Falling back to CPU."
                );
                #[cfg(any(feature = "cuda", feature = "metal"))]
                warn!("No accessible GPU device found; falling back to CPU");

                Ok((Device::Cpu, DeviceType::Cpu, "CPU fallback".to_string()))
            }
        }
    }

    /// Quick runtime check for a specific GPU backend.
    pub fn backend_available(backend: GpuBackend, _index: usize) -> bool {
        match backend {
            GpuBackend::Auto => false,
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda => Device::new_cuda(_index).is_ok(),
            #[cfg(feature = "metal")]
            GpuBackend::Metal => Device::new_metal(_index).is_ok(),
            _ => false,
        }
    }
}

impl Trainer for GpuTrainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        self.inner.train(batch)
    }

    fn train_with_gradients(&self, batch: &TrainingBatch) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        self.inner.train_with_gradients(batch)
    }

    fn train_genome(&self, msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        self.inner.train_genome(msg)
    }

    fn train_genome_with_gradients(&self, msg: &GenomeTrainingBatchMsg) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        self.inner.train_genome_with_gradients(msg)
    }

    fn device_info(&self) -> DeviceInfo {
        let mut info = DeviceInfo {
            device_type: self.device_type.clone(),
            name: self.device_name.clone(),
            threads: self.threads,
            ..Default::default()
        };

        if self.device_type == DeviceType::Cuda {
            if let Ok(stats) = GpuMonitor::get_stats(self.device_index as u32) {
                info.memory_used = Some(stats.memory_used);
                info.temperature = Some(stats.temperature);
                info.utilization = Some(stats.utilization);
            }
        }

        info
    }
}
