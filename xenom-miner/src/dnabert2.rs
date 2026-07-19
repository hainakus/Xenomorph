use candle_core::{DType, Device, Module, ModuleT, Result as CandleResult, Tensor};
use candle_nn::{embedding, layer_norm, linear, linear_no_bias, Activation, Dropout, Embedding, LayerNorm, Linear, VarBuilder, VarMap};

use crate::model::DnaBert2Config;

/// DNABERT-2 / MosaicBERT embeddings: word + token_type, no positional embeddings (ALiBi).
struct DnaBert2Embeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    pad_token_id: u32,
}

impl DnaBert2Embeddings {
    fn new(vb: VarBuilder, config: &DnaBert2Config) -> CandleResult<Self> {
        let word_embeddings = embedding(config.vocab_size, config.hidden_size, vb.pp("word_embeddings"))?;
        let token_type_embeddings = embedding(config.type_vocab_size, config.hidden_size, vb.pp("token_type_embeddings"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("layer_norm"))?;
        let dropout = Dropout::new(config.hidden_dropout);
        Ok(Self { word_embeddings, token_type_embeddings, layer_norm, dropout, pad_token_id: config.pad_token_id })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: Option<&Tensor>) -> CandleResult<Tensor> {
        let words = self.word_embeddings.forward(input_ids)?;

        let token_type_ids = match token_type_ids {
            Some(ids) => ids.clone(),
            None => Tensor::zeros(input_ids.dims(), input_ids.dtype(), input_ids.device())?,
        };
        let token_types = self.token_type_embeddings.forward(&token_type_ids)?;

        let embeddings = words.broadcast_add(&token_types)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward_t(&embeddings, false)
    }
}

/// ALiBi slope computation per head.
struct AlibiSlopes {
    slopes: Tensor,
}

impl AlibiSlopes {
    fn new(num_heads: usize, device: &Device) -> CandleResult<Self> {
        let slopes = Self::compute_slopes(num_heads);
        let slopes = Tensor::new(slopes.as_slice(), device)?.reshape((num_heads, 1, 1))?;
        Ok(Self { slopes })
    }

    fn compute_slopes(n_heads: usize) -> Vec<f32> {
        fn power_of_2(n: usize) -> Vec<f32> {
            let start = 2f64.powf(-(2f64.powf(-(f64::log2(n as f64) - 3.0)))) as f32;
            let ratio = start;
            (0..n).map(|i| start * ratio.powi(i as i32)).collect()
        }

        fn slopes(n: usize) -> Vec<f32> {
            if f64::log2(n as f64).fract() == 0.0 {
                return power_of_2(n);
            }
            let closest = 2usize.pow(f64::log2(n as f64).floor() as u32);
            let mut a = power_of_2(closest);
            let b = slopes(2 * closest);
            a.extend(b.into_iter().step_by(2).take(n - closest));
            a
        }

        slopes(n_heads)
    }

    /// Build an ALiBi bias tensor of shape `[1, num_heads, seq_len, seq_len]`.
    fn bias(&self, seq_len: usize) -> CandleResult<Tensor> {
        let device = self.slopes.device();
        let context = Tensor::arange(0f32, seq_len as f32, device)?.reshape((seq_len, 1))?;
        let memory = Tensor::arange(0f32, seq_len as f32, device)?.reshape((1, seq_len))?;
        let relative = context.broadcast_sub(&memory)?.abs()?; // [seq, seq]
        let relative = relative.unsqueeze(0)?; // [1, seq, seq]
        self.slopes.broadcast_mul(&relative.neg()?)
    }
}

/// DNABERT-2 self-attention with separate Q/K/V projections and ALiBi.
struct DnaBert2SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    alibi: AlibiSlopes,
}

impl DnaBert2SelfAttention {
    fn new(vb: VarBuilder, config: &DnaBert2Config, device: &Device) -> CandleResult<Self> {
        let all_head_size = config.hidden_size;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let query = linear(config.hidden_size, all_head_size, vb.pp("query"))?;
        let key = linear(config.hidden_size, all_head_size, vb.pp("key"))?;
        let value = linear(config.hidden_size, all_head_size, vb.pp("value"))?;
        let dropout = Dropout::new(config.attention_dropout);
        let alibi = AlibiSlopes::new(config.num_attention_heads, device)?;

        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            alibi,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor, batch: usize, seq: usize) -> CandleResult<Tensor> {
        xs.reshape((batch, seq, self.num_attention_heads, self.attention_head_size))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let dims = hidden_states.dims();
        let (batch, seq, _) = (dims[0], dims[1], dims[2]);

        let mixed_query = self.query.forward(hidden_states)?;
        let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?, batch, seq)?;
        let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?, batch, seq)?;
        let query_layer = self.transpose_for_scores(&mixed_query, batch, seq)?;

        let scale = 1.0 / (self.attention_head_size as f64).sqrt();
        let scale_t = Tensor::new(scale as f32, hidden_states.device())?;

        let key_t = key_layer.transpose(2, 3)?.contiguous()?;
        let mut attention_scores = query_layer.matmul(&key_t)?;
        attention_scores = attention_scores.broadcast_mul(&scale_t)?;

        let alibi = self.alibi.bias(seq)?.unsqueeze(0)?; // [1, heads, seq, seq]
        attention_scores = attention_scores.broadcast_add(&alibi)?;
        attention_scores = attention_scores.broadcast_add(attention_mask)?;

        let attention_probs = candle_nn::ops::softmax(&attention_scores, 3)?;
        let attention_probs = self.dropout.forward_t(&attention_probs, false)?;

        let attention_output = attention_probs.matmul(&value_layer)?;
        attention_output.transpose(1, 2)?.reshape((batch, seq, self.all_head_size))
    }
}

/// Post-attention dense + residual LayerNorm.
struct DnaBert2SelfOutput {
    dense: Linear,
    dropout: Dropout,
    layer_norm: LayerNorm,
}

impl DnaBert2SelfOutput {
    fn new(vb: VarBuilder, config: &DnaBert2Config) -> CandleResult<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("layer_norm"))?;
        let dropout = Dropout::new(config.hidden_dropout);
        Ok(Self { dense, dropout, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> CandleResult<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward_t(&hidden_states, false)?;
        self.layer_norm.forward(&(hidden_states.broadcast_add(input_tensor)?))
    }
}

/// Attention sub-layer (self-attention + output).
struct DnaBert2Attention {
    self_attn: DnaBert2SelfAttention,
    output: DnaBert2SelfOutput,
}

impl DnaBert2Attention {
    fn new(vb: VarBuilder, config: &DnaBert2Config, device: &Device) -> CandleResult<Self> {
        Ok(Self {
            self_attn: DnaBert2SelfAttention::new(vb.pp("self"), config, device)?,
            output: DnaBert2SelfOutput::new(vb.pp("output"), config)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let self_output = self.self_attn.forward(hidden_states, attention_mask)?;
        self.output.forward(&self_output, hidden_states)
    }
}

/// Gated GeLU MLP as in MosaicBERT.
struct DnaBert2GatedMlp {
    up_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
    layer_norm: LayerNorm,
    activation: Activation,
    intermediate_size: usize,
}

impl DnaBert2GatedMlp {
    fn new(vb: VarBuilder, config: &DnaBert2Config) -> CandleResult<Self> {
        let up_proj = linear_no_bias(config.hidden_size, config.intermediate_size * 2, vb.pp("up_proj"))?;
        let down_proj = linear(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("layer_norm"))?;
        let dropout = Dropout::new(config.hidden_dropout);
        let activation = Activation::Gelu;
        Ok(Self { up_proj, down_proj, dropout, layer_norm, activation, intermediate_size: config.intermediate_size })
    }

    fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.up_proj.forward(hidden_states)?;

        let rank = hidden_states.dims().len();
        let gated = hidden_states.narrow(rank - 1, 0, self.intermediate_size)?;
        let non_gated = hidden_states.narrow(rank - 1, self.intermediate_size, self.intermediate_size)?;
        let hidden_states = self.activation.forward(&gated)?.broadcast_mul(&non_gated)?;

        let hidden_states = self.dropout.forward_t(&hidden_states, false)?;
        let hidden_states = self.down_proj.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states.broadcast_add(&residual)?))
    }
}

/// One DNABERT-2 transformer layer.
struct DnaBert2Layer {
    attention: DnaBert2Attention,
    mlp: DnaBert2GatedMlp,
}

impl DnaBert2Layer {
    fn new(vb: VarBuilder, config: &DnaBert2Config, device: &Device) -> CandleResult<Self> {
        Ok(Self {
            attention: DnaBert2Attention::new(vb.pp("attention"), config, device)?,
            mlp: DnaBert2GatedMlp::new(vb.pp("mlp"), config)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        self.mlp.forward(&attention_output)
    }
}

/// Stack of DNABERT-2 layers.
struct DnaBert2Encoder {
    layers: Vec<DnaBert2Layer>,
}

impl DnaBert2Encoder {
    fn new(vb: VarBuilder, config: &DnaBert2Config, device: &Device) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DnaBert2Layer::new(vb.pp(format!("layer.{}", i)), config, device)?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

/// Masked LM head: transform (dense + gelu + LN) then decoder tied to word embeddings.
struct DnaBert2LMPredictionHead {
    transform_dense: Linear,
    transform_layer_norm: LayerNorm,
    transform_act: Activation,
    decoder: Linear,
}

impl DnaBert2LMPredictionHead {
    fn new(vb: VarBuilder, config: &DnaBert2Config, word_embeddings: &Embedding) -> CandleResult<Self> {
        let lm_vb = vb.pp("lm_head");
        let transform_dense = linear(config.hidden_size, config.hidden_size, lm_vb.pp("transform").pp("dense"))?;
        let transform_layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, lm_vb.pp("transform").pp("layer_norm"))?;
        let decoder_weight = word_embeddings.embeddings().clone();
        let decoder_bias = lm_vb.get(config.vocab_size, "bias")?;
        let decoder = Linear::new(decoder_weight, Some(decoder_bias));
        Ok(Self {
            transform_dense,
            transform_layer_norm,
            transform_act: Activation::Gelu,
            decoder,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let hidden_states = self.transform_dense.forward(hidden_states)?;
        let hidden_states = self.transform_act.forward(&hidden_states)?;
        let hidden_states = self.transform_layer_norm.forward(&hidden_states)?;
        self.decoder.forward(&hidden_states)
    }
}

/// DNABERT-2 base model (embeddings + encoder).
pub struct DnaBert2Model {
    embeddings: DnaBert2Embeddings,
    encoder: DnaBert2Encoder,
    pub config: DnaBert2Config,
}

impl DnaBert2Model {
    fn new(vb: VarBuilder, config: DnaBert2Config, device: &Device) -> CandleResult<Self> {
        let embeddings = DnaBert2Embeddings::new(vb.pp("model").pp("embeddings"), &config)?;
        let encoder = DnaBert2Encoder::new(vb.pp("model").pp("encoder"), &config, device)?;
        Ok(Self { embeddings, encoder, config })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let dims = input_ids.dims();
        let (_batch, _seq) = (dims[0], dims[1]);

        let attention_mask = match attention_mask {
            Some(mask) => mask.to_dtype(DType::F32)?,
            None => input_ids.ne(self.embeddings.pad_token_id as f64)?.to_dtype(DType::F32)?,
        };

        let ones = Tensor::ones(attention_mask.dims(), DType::F32, attention_mask.device())?;
        let additive = ones.broadcast_sub(&attention_mask)?;
        let scale = Tensor::new(-10000.0f32, attention_mask.device())?;
        let additive = additive.broadcast_mul(&scale)?;
        let additive = additive.unsqueeze(1)?.unsqueeze(1)?; // [batch, 1, 1, seq]

        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output, &additive)?;
        Ok(sequence_output)
    }
}

/// DNABERT-2 model with masked language modeling head.
pub struct DnaBert2ForMaskedLM {
    model: DnaBert2Model,
    lm_head: DnaBert2LMPredictionHead,
}

impl DnaBert2ForMaskedLM {
    /// Build a model from an existing `VarBuilder` (e.g. `VarMap` for training or
    /// `from_buffered_safetensors` for inference).
    pub fn new(vb: VarBuilder, config: DnaBert2Config, device: &Device) -> CandleResult<Self> {
        let model = DnaBert2Model::new(vb.clone(), config, device)?;
        let lm_head = DnaBert2LMPredictionHead::new(vb, &model.config, &model.embeddings.word_embeddings)?;
        Ok(Self { model, lm_head })
    }

    /// Load a `DnaBert2ForMaskedLM` from raw `model.safetensors` bytes (inference only).
    pub fn load(config: DnaBert2Config, weights: Vec<u8>, dtype: DType, device: &Device) -> CandleResult<Self> {
        let vb = VarBuilder::from_buffered_safetensors(weights, dtype, device)?;
        Self::new(vb, config, device)
    }

    /// Load a trainable model from raw `model.safetensors` bytes, returning it together with
    /// the underlying `VarMap` so that an optimizer can be created.
    pub fn load_for_training(config: DnaBert2Config, weights: Vec<u8>, dtype: DType, device: &Device) -> CandleResult<(Self, VarMap)> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        let model = Self::new(vb, config, device)?;
        let tensors = candle_core::safetensors::load_buffer(&weights, device)?;
        {
            let tensor_data = varmap.data().lock().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            for (name, tensor) in tensors.iter() {
                if let Some(var) = tensor_data.get(name) {
                    var.set(tensor)?;
                }
                // Unknown keys are ignored so that tied/decoded weights do not break loading.
            }
        }
        Ok((model, varmap))
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, token_type_ids, attention_mask)?;
        self.lm_head.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{safetensors, Tensor};
    use std::collections::HashMap;

    fn build_tiny_config() -> DnaBert2Config {
        DnaBert2Config {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 8,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".to_string(),
            position_embedding_type: "alibi".to_string(),
            alibi_starting_size: Some(16),
            tie_word_embeddings: true,
            pad_token_id: 0,
            mask_token_id: 4,
            bos_token_id: 1,
            eos_token_id: 2,
            num_labels: None,
        }
    }

    fn insert_weight(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize], device: &Device) {
        let n = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() + 0.001).collect();
        let t = Tensor::from_vec(data, shape, device).unwrap();
        map.insert(name.to_string(), t);
    }

    fn build_tiny_safetensors() -> Vec<u8> {
        let device = Device::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        let config = build_tiny_config();
        insert_weight(&mut tensors, "model.embeddings.word_embeddings.weight", &[config.vocab_size, config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.token_type_embeddings.weight", &[config.type_vocab_size, config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.weight", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.bias", &[config.hidden_size], &device);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.encoder.layer.{}", i);
            insert_weight(&mut tensors, &format!("{}.attention.self.query.weight", prefix), &[config.hidden_size, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.query.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.key.weight", prefix), &[config.hidden_size, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.key.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.value.weight", prefix), &[config.hidden_size, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.self.value.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.dense.weight", prefix), &[config.hidden_size, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.dense.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.weight", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.bias", prefix), &[config.hidden_size], &device);

            insert_weight(&mut tensors, &format!("{}.mlp.up_proj.weight", prefix), &[config.intermediate_size * 2, config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.down_proj.weight", prefix), &[config.hidden_size, config.intermediate_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.down_proj.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.layer_norm.weight", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.layer_norm.bias", prefix), &[config.hidden_size], &device);
        }

        insert_weight(&mut tensors, "lm_head.transform.dense.weight", &[config.hidden_size, config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.dense.bias", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.layer_norm.weight", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.layer_norm.bias", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.bias", &[config.vocab_size], &device);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");
        safetensors::save(&tensors, &path).unwrap();
        std::fs::read(&path).unwrap()
    }

    #[test]
    fn test_alibi_slopes() {
        let slopes = AlibiSlopes::compute_slopes(12);
        assert_eq!(slopes.len(), 12);
        assert!(slopes[0] > slopes[11]);
    }

    #[test]
    fn test_forward_shapes() {
        let device = Device::Cpu;
        let weights = build_tiny_safetensors();
        let config = build_tiny_config();
        let model = DnaBert2ForMaskedLM::load(config, weights, DType::F32, &device).unwrap();

        let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).unwrap();
        let attention_mask = Tensor::new(&[[1u32, 1, 1, 1, 1]], &device).unwrap();
        let logits = model.forward(&input_ids, None, Some(&attention_mask)).unwrap();

        assert_eq!(logits.dims().to_vec(), vec![1, 5, 8]);
    }
}
