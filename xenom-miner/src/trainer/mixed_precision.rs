//! Mixed-precision training helpers (FP16 forward/backward with FP32 optimizer state).

use candle_core::{DType, Result as CandleResult, Tensor};
use tracing::{info, warn};

/// A simple loss scaler for FP16 training.
///
/// Gradients are computed against a scaled loss so small values do not underflow
/// in FP16. The same scale factor is later applied to the learning rate instead
/// of unscaling the gradients, which is equivalent and avoids mutating the
/// immutable `GradStore`.
#[derive(Debug, Clone, Copy)]
pub struct MixedPrecisionScaler {
    /// Current loss scale.
    scale: f32,
    /// Whether FP16 is enabled.
    enabled: bool,
}

impl MixedPrecisionScaler {
    /// Default scale factor (2^16). Large enough to keep small gradients visible
    /// in FP16, small enough to avoid frequent overflow on DNABERT-2 sized models.
    pub const DEFAULT_SCALE: f32 = 65536.0;

    /// Create a new scaler.
    pub fn new(enabled: bool) -> Self {
        Self { scale: Self::DEFAULT_SCALE, enabled }
    }

    /// Return the dtype used for forward/backward computations.
    pub fn compute_dtype(&self) -> DType {
        if self.enabled {
            DType::F16
        } else {
            DType::F32
        }
    }

    /// Scale a loss tensor before backward.
    pub fn scale_loss(&self, loss: &Tensor) -> CandleResult<Tensor> {
        if !self.enabled {
            return Ok(loss.clone());
        }
        loss * (self.scale as f64)
    }

    /// Adjust an external learning rate to account for the loss scale.
    /// The optimizer sees scaled gradients, so the effective LR is `lr / scale`.
    pub fn effective_learning_rate(&self, base_lr: f32) -> f32 {
        if self.enabled {
            base_lr / self.scale
        } else {
            base_lr
        }
    }

    /// Check whether any gradient contains inf/nan. If so, the step should be
    /// skipped and the scale reduced.
    pub fn has_overflow(&self, gradients: &[(String, Tensor)]) -> CandleResult<bool> {
        if !self.enabled {
            return Ok(false);
        }
        for (_, g) in gradients.iter() {
            let flat = g.flatten_all()?;
            let values = flat.to_vec1::<f32>()?;
            for v in values.iter() {
                if v.is_nan() || v.is_infinite() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Update the loss scale based on whether overflow occurred.
    pub fn update_scale(&mut self, had_overflow: bool) {
        if !self.enabled {
            return;
        }
        if had_overflow {
            self.scale /= 2.0;
            warn!("FP16 gradient overflow detected; reduced loss scale to {}", self.scale);
        } else if self.scale < Self::DEFAULT_SCALE {
            // Gradually recover, but cap at the default.
            self.scale = (self.scale * 1.1).min(Self::DEFAULT_SCALE);
            info!("No FP16 overflow; increased loss scale to {}", self.scale);
        }
    }

    /// Return the current scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

/// Convert a tensor to a dtype that is safe for CPU-side gradient averaging.
/// FP16 gradients are converted to FP32 to keep the average stable across GPUs.
pub fn to_grad_dtype(tensor: &Tensor) -> CandleResult<Tensor> {
    if tensor.dtype() == DType::F16 {
        tensor.to_dtype(DType::F32)
    } else {
        Ok(tensor.clone())
    }
}
