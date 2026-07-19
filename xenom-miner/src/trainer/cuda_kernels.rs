use anyhow::{bail, Result};
use candle_core::{DType, Tensor};

/// Placeholder for a custom CUDA ALiBi attention bias kernel.
///
/// Candle already implements ALiBi via its standard ops, so this is kept as an
/// extension point for future kernel-level optimizations.
pub fn compute_alibi_bias(
    _num_heads: usize,
    _seq_len: usize,
    _device: &candle_core::Device,
) -> Result<Tensor> {
    bail!("Custom CUDA ALiBi kernel is not yet implemented; use the default Candle path")
}

/// Suggested dtype for custom kernels: F32 for numerical stability.
pub fn kernel_dtype() -> DType {
    DType::F32
}
