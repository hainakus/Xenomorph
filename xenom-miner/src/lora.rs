use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, VarMap};

/// Configuration for Low-Rank Adaptation (LoRA) fine-tuning.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition.
    pub rank: usize,
    /// LoRA scaling factor. The adapter output is multiplied by `alpha / rank`.
    pub alpha: f32,
    /// Dropout probability applied to the adapter output (currently unused).
    pub dropout: f32,
    /// Names of Linear module groups that should receive LoRA adapters.
    /// A module is targeted if its last path segment or its `"{second_last}_{last}"`
    /// suffix is in this set.
    pub target_modules: HashSet<String>,
}

impl LoraConfig {
    /// Default LoRA target modules for DNABERT-2.
    pub fn default_target_modules() -> HashSet<String> {
        ["query", "key", "value", "transform_dense", "up_proj", "down_proj"].iter().map(|s| s.to_string()).collect()
    }

    /// Parse LoRA configuration from environment variables.
    ///
    /// LoRA is enabled by default. Set `XENO_LORA` to `0`, `false`, `no`, or `off`
    /// to disable it. Other variables override the defaults:
    /// - `XENO_LORA_RANK` (default 8)
    /// - `XENO_LORA_ALPHA` (default 16.0)
    /// - `XENO_LORA_DROPOUT` (default 0.0)
    /// - `XENO_LORA_TARGET_MODULES` comma-separated (default `query,key,value,transform_dense,up_proj,down_proj`)
    pub fn from_env() -> Option<Self> {
        let enabled = match std::env::var("XENO_LORA").ok().as_deref() {
            Some("0") | Some("false") | Some("no") | Some("off") => false,
            _ => true,
        };
        if !enabled {
            return None;
        }

        let rank = std::env::var("XENO_LORA_RANK").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let alpha = std::env::var("XENO_LORA_ALPHA").ok().and_then(|v| v.parse().ok()).unwrap_or(16.0);
        let dropout = std::env::var("XENO_LORA_DROPOUT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let target_modules = std::env::var("XENO_LORA_TARGET_MODULES")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(Self::default_target_modules);

        Some(Self { rank, alpha, dropout, target_modules })
    }
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self { rank: 8, alpha: 16.0, dropout: 0.0, target_modules: Self::default_target_modules() }
    }
}

/// A `Linear` layer augmented with a low-rank adapter.
///
/// The effective weight is `W = W0 + (alpha / rank) * B * A`, where `A` has shape
/// `[rank, in_features]`, `B` has shape `[out_features, rank]`, and `W0` is the
/// frozen base weight.
#[derive(Clone, Debug)]
pub struct LoraLinear {
    base: Linear,
    lora_a: Tensor,
    lora_b: Tensor,
    scale: f64,
}

impl LoraLinear {
    pub fn new(base: Linear, lora_a: Tensor, lora_b: Tensor, scale: f64) -> Self {
        Self { base, lora_a, lora_b, scale }
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let base = self.base.forward(x)?;

        let lora_a_t = match *x.dims() {
            [b1, b2, _, _] => self.lora_a.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.lora_a.broadcast_left(bsize)?.t()?,
            _ => self.lora_a.t()?,
        };
        let lora_b_t = match *x.dims() {
            [b1, b2, _, _] => self.lora_b.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.lora_b.broadcast_left(bsize)?.t()?,
            _ => self.lora_b.t()?,
        };

        let lora = x.matmul(&lora_a_t)?.matmul(&lora_b_t)?;
        let scale_t = Tensor::new(self.scale as f32, x.device())?.to_dtype(x.dtype())?;
        base.broadcast_add(&lora.broadcast_mul(&scale_t)?)
    }
}

/// Either a standard `Linear` layer or a LoRA-augmented one.
#[derive(Clone, Debug)]
pub enum LinearLayer {
    Standard(Linear),
    Lora(LoraLinear),
}

impl Module for LinearLayer {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Standard(l) => l.forward(x),
            Self::Lora(l) => l.forward(x),
        }
    }
}

/// Builder that constructs DNABERT-2 modules either from a standard `VarBuilder`
/// (full fine-tuning) or from frozen base weights plus trainable LoRA parameters.
#[derive(Clone)]
pub struct ModelBuilder {
    varmap: Arc<VarMap>,
    base_weights: Arc<HashMap<String, Tensor>>,
    lora_config: Option<Arc<LoraConfig>>,
    path: Vec<String>,
    dtype: DType,
    device: Device,
}

impl ModelBuilder {
    pub fn new(
        varmap: Arc<VarMap>,
        base_weights: Arc<HashMap<String, Tensor>>,
        lora_config: Option<LoraConfig>,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self { varmap, base_weights, lora_config: lora_config.map(Arc::new), path: Vec::new(), dtype, device }
    }

    /// Return a `VarBuilder` positioned at the current path.
    fn var_builder(&self) -> VarBuilder<'_> {
        let mut vb = VarBuilder::from_varmap(&self.varmap, self.dtype, &self.device);
        for p in &self.path {
            vb = vb.pp(p);
        }
        vb
    }

    /// Return a new builder rooted at `s` under the current path.
    pub fn pp(&self, s: impl ToString) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            varmap: self.varmap.clone(),
            base_weights: self.base_weights.clone(),
            lora_config: self.lora_config.clone(),
            path,
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }

    /// Return the LoRA configuration, if any.
    pub fn lora_config(&self) -> Option<&LoraConfig> {
        self.lora_config.as_deref()
    }

    fn is_lora(&self) -> bool {
        self.lora_config.is_some()
    }

    /// Check whether the current path corresponds to a targeted Linear module.
    fn is_targeted(&self) -> bool {
        let cfg = match &self.lora_config {
            Some(cfg) => cfg,
            None => return false,
        };
        let Some(last) = self.path.last() else {
            return false;
        };
        if cfg.target_modules.contains(last) {
            return true;
        }
        if self.path.len() >= 2 {
            let combined = format!("{}_{}", self.path[self.path.len() - 2], last);
            if cfg.target_modules.contains(&combined) {
                return true;
            }
        }
        false
    }

    fn full_key(&self, suffix: &str) -> String {
        if self.path.is_empty() {
            suffix.to_string()
        } else {
            format!("{}.{}", self.path.join("."), suffix)
        }
    }

    fn get_base_tensor(&self, suffix: &str) -> CandleResult<Tensor> {
        let key = self.full_key(suffix);
        let tensor = self.base_weights.get(&key).ok_or_else(|| candle_core::Error::Msg(format!("Missing base weight: {}", key)))?;
        tensor.to_device(&self.device)?.to_dtype(self.dtype)
    }

    /// Retrieve a single tensor (e.g. `lm_head.bias`) as a constant in LoRA mode
    /// or as a `VarBuilder` variable otherwise.
    pub fn get<S: Into<candle_core::Shape>>(&self, shape: S, name: &str) -> CandleResult<Tensor> {
        if self.is_lora() {
            let _shape = shape.into();
            self.get_base_tensor(name)
        } else {
            self.var_builder().get(shape, name)
        }
    }

    /// Build an `Embedding` layer.
    pub fn embedding(&self, num_embeddings: usize, hidden_size: usize) -> CandleResult<Embedding> {
        if self.is_lora() {
            let weight = self.get_base_tensor("weight")?;
            Ok(Embedding::new(weight, hidden_size))
        } else {
            candle_nn::embedding(num_embeddings, hidden_size, self.var_builder())
        }
    }

    /// Build a `LayerNorm` layer.
    pub fn layer_norm(&self, size: usize, eps: f64) -> CandleResult<LayerNorm> {
        if self.is_lora() {
            let weight = self.get_base_tensor("weight")?;
            let bias = self.get_base_tensor("bias")?;
            Ok(LayerNorm::new(weight, bias, eps))
        } else {
            candle_nn::layer_norm(size, eps, self.var_builder())
        }
    }

    /// Build a `Linear` layer, optionally with LoRA.
    pub fn linear(&self, in_features: usize, out_features: usize) -> CandleResult<LinearLayer> {
        self.build_linear(in_features, out_features, true)
    }

    /// Build a `Linear` layer without bias, optionally with LoRA.
    pub fn linear_no_bias(&self, in_features: usize, out_features: usize) -> CandleResult<LinearLayer> {
        self.build_linear(in_features, out_features, false)
    }

    fn build_linear(&self, in_features: usize, out_features: usize, bias: bool) -> CandleResult<LinearLayer> {
        if self.is_lora() {
            let weight = self.get_base_tensor("weight")?;
            let bias_tensor = if bias { Some(self.get_base_tensor("bias")?) } else { None };
            let base = Linear::new(weight, bias_tensor);

            if self.is_targeted() {
                let cfg = self.lora_config.as_ref().unwrap();
                let rank = cfg.rank;
                let vb = self.var_builder();
                let lora_a = vb.get_with_hints((rank, in_features), "lora_a", candle_nn::init::DEFAULT_KAIMING_UNIFORM)?;
                let lora_b = vb.get_with_hints((out_features, rank), "lora_b", candle_nn::init::ZERO)?;
                let scale = cfg.alpha as f64 / rank as f64;
                Ok(LinearLayer::Lora(LoraLinear::new(base, lora_a, lora_b, scale)))
            } else {
                Ok(LinearLayer::Standard(base))
            }
        } else {
            let vb = self.var_builder();
            if bias {
                Ok(LinearLayer::Standard(candle_nn::linear(in_features, out_features, vb)?))
            } else {
                Ok(LinearLayer::Standard(candle_nn::linear_no_bias(in_features, out_features, vb)?))
            }
        }
    }
}
