//! Gradient-checkpointed forward pass for the DNABERT-2 encoder.
//!
//! This module is currently a structural placeholder. Real gradient checkpointing
//! for Candle transformer blocks requires segmenting the encoder into checkpoint
//! groups, saving segment inputs, and recomputing segment forwards during backward.
//! That implementation is left as a follow-up once the core multi-GPU data-parallel
//! trainer is stable.

use anyhow::Result;
use candle_core::Tensor;

/// Trait for models that can optionally run with gradient checkpointing.
pub trait CheckpointedForward {
    /// Forward pass with gradient checkpointing enabled.
    ///
    /// The default implementation delegates to the normal forward pass; this is
    /// the safe fallback until checkpointing is wired into `DnaBert2Model`.
    fn forward_with_checkpointing(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor>;
}

/// Informational marker that gradient checkpointing was requested.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientCheckpointing {
    Disabled,
    Enabled,
}

impl GradientCheckpointing {
    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled)
    }
}

impl From<bool> for GradientCheckpointing {
    fn from(value: bool) -> Self {
        if value {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

/// Stub: no-op checkpointed forward.
///
/// When this is wired into `DnaBert2Model`, it should:
/// 1. Split `encoder.layers` into segments of N layers.
/// 2. For each segment, run forward, optionally detach/save the segment input,
///    and keep only the output for the next segment.
/// 3. During backward, re-run the segment forward from the saved input, call
///    backward on the segment output, and propagate the input gradient.
pub fn checkpointed_forward_stub(_num_layers: usize) -> Result<()> {
    // Placeholder: real checkpointing will be implemented here.
    Ok(())
}
