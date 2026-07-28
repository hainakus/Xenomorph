//! Mini Genome Model (MGM-1)
//! Transformer pequeno (~1-2M parametros) para MLM em sequencias de DNA
//! Otimizado para treinamento rapido em CPU/GPU via Candle

fn default_label_smoothing() -> f64 {
    0.1
}

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Dropout, Embedding, Init, LayerNorm, Linear, Optimizer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use candle_nn::var_builder::VarBuilder;
pub use candle_nn::var_map::VarMap;

/// Configuracao do MGM-1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniGenomeConfig {
    /// Tamanho do vocabulario (A, C, G, T, [MASK], [PAD], [CLS], [SEP])
    pub vocab_size: usize,
    /// Dimensao dos embeddings
    pub d_model: usize,
    /// Numero de cabecas de atencao
    pub n_heads: usize,
    /// Numero de camadas do transformer
    pub n_layers: usize,
    /// Dimensao da feed-forward network (tipicamente 4x d_model)
    pub d_ff: usize,
    /// Tamanho maximo da sequencia
    pub max_seq_len: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Label smoothing for MLM cross-entropy (0 = hard targets).  A small value
    /// (e.g. 0.1) prevents the model from becoming overconfident and collapsing
    /// to a single nucleotide prediction.
    #[serde(default = "default_label_smoothing")]
    pub label_smoothing: f64,
}

impl Default for MiniGenomeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8,
            d_model: 128,
            n_heads: 4,
            n_layers: 4,
            d_ff: 512,
            max_seq_len: 512,
            dropout: 0.1,
            label_smoothing: default_label_smoothing(),
        }
    }
}

impl MiniGenomeConfig {
    /// Configuracao ultra-pequena para testes rapidos
    pub fn tiny() -> Self {
        Self { vocab_size: 8, d_model: 64, n_heads: 2, n_layers: 2, d_ff: 256, max_seq_len: 256, dropout: 0.1, label_smoothing: 0.0 }
    }

    /// Numero total de parametros (estimativa)
    pub fn num_parameters(&self) -> usize {
        let embedding = self.vocab_size * self.d_model;
        let attention = self.n_layers * (4 * self.d_model * self.d_model);
        let ffn = self.n_layers * (self.d_model * self.d_ff * 2);
        let layer_norm = self.n_layers * 2 * self.d_model + 2 * self.d_model;
        let head = self.d_model * self.vocab_size;

        embedding + attention + ffn + layer_norm + head
    }
}

/// Tokenizador simples para DNA
pub struct DnaTokenizer {
    /// Mapeia char -> token ID
    char_to_id: HashMap<char, usize>,
    /// Mapeia token ID -> char
    id_to_char: HashMap<usize, char>,
}

impl DnaTokenizer {
    pub fn new() -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // Tokens especiais primeiro
        let tokens = [
            ('A', 0),
            ('C', 1),
            ('G', 2),
            ('T', 3),
            ('[', 4), // [MASK]
            (' ', 5), // [PAD]
            (']', 6), // [CLS]
            ('|', 7), // [SEP]
        ];

        for (ch, id) in tokens {
            char_to_id.insert(ch, id);
            id_to_char.insert(id, ch);
        }

        Self { char_to_id, id_to_char }
    }

    /// Tokeniza uma sequencia de DNA
    pub fn encode(&self, sequence: &str) -> Vec<usize> {
        sequence.chars().filter_map(|c| self.char_to_id.get(&c).copied()).collect()
    }

    /// Decodifica tokens para string
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter().filter_map(|&id| self.id_to_char.get(&id).copied()).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.char_to_id.len()
    }
}

impl Default for DnaTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Embedding posicional sinusoidal (como no Transformer original)
pub struct SinusoidalPositionalEmbedding {
    d_model: usize,
}

impl SinusoidalPositionalEmbedding {
    pub fn new(_max_seq_len: usize, d_model: usize) -> Self {
        Self { d_model }
    }

    pub fn forward(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mut pe = vec![0.0f32; seq_len * self.d_model];

        for pos in 0..seq_len {
            for i in 0..self.d_model {
                let angle = pos as f32 / f32::powf(10000.0, (2 * (i / 2)) as f32 / self.d_model as f32);
                pe[pos * self.d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Tensor::from_vec(pe, (seq_len, self.d_model), device)
    }
}

/// Multi-Head Attention simplificado
pub struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    n_heads: usize,
    d_k: usize,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder, d_model: usize, n_heads: usize, dropout: f64) -> Result<Self> {
        assert_eq!(d_model % n_heads, 0, "d_model deve ser divisivel por n_heads");

        let d_k = d_model / n_heads;

        let w_q = linear(d_model, d_model, vb.pp("w_q"))?;
        let w_k = linear(d_model, d_model, vb.pp("w_k"))?;
        let w_v = linear(d_model, d_model, vb.pp("w_v"))?;
        let w_o = linear(d_model, d_model, vb.pp("w_o"))?;

        Ok(Self { w_q, w_k, w_v, w_o, n_heads, d_k, dropout: Dropout::new(dropout as f32) })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, d_model) = xs.dims3()?;

        // Projecoes lineares
        let q = self.w_q.forward(xs)?;
        let k = self.w_k.forward(xs)?;
        let v = self.w_v.forward(xs)?;

        // Reshape para multi-head: (batch, n_heads, seq_len, d_k)
        let q = q.reshape((batch, seq_len, self.n_heads, self.d_k))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((batch, seq_len, self.n_heads, self.d_k))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((batch, seq_len, self.n_heads, self.d_k))?.transpose(1, 2)?.contiguous()?;

        // Atencao escalada: Q @ K^T / sqrt(d_k)
        let scale = Tensor::new((self.d_k as f32).sqrt(), q.device())?;
        let kt = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&kt)?.broadcast_div(&scale)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let attn_weights = self.dropout.forward(&attn_weights, false)?;

        // Aplicar atencao aos valores
        let attn_output = attn_weights.matmul(&v)?;

        // Concatenar cabecas e projetar
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((batch, seq_len, d_model))?;

        self.w_o.forward(&attn_output)
    }
}

/// Feed-Forward Network
pub struct FeedForward {
    w1: Linear,
    w2: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, d_model: usize, d_ff: usize, dropout: f64) -> Result<Self> {
        let w1 = linear(d_model, d_ff, vb.pp("w1"))?;
        let w2 = linear(d_ff, d_model, vb.pp("w2"))?;

        Ok(Self { w1, w2, dropout: Dropout::new(dropout as f32) })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.w1.forward(x)?.relu()?;
        let hidden = self.dropout.forward(&hidden, false)?;
        self.w2.forward(&hidden)
    }
}

/// Bloco Transformer completo
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(vb: VarBuilder, config: &MiniGenomeConfig) -> Result<Self> {
        let attention = MultiHeadAttention::new(vb.pp("attention"), config.d_model, config.n_heads, config.dropout)?;

        let ffn = FeedForward::new(vb.pp("ffn"), config.d_model, config.d_ff, config.dropout)?;

        let norm1 = layer_norm(config.d_model, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(config.d_model, 1e-5, vb.pp("norm2"))?;

        Ok(Self { attention, ffn, norm1, norm2, dropout: Dropout::new(config.dropout as f32) })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm architecture
        let attn_output = self.attention.forward(&self.norm1.forward(x)?)?;
        let x = x.add(&attn_output)?;
        let x = self.dropout.forward(&x, false)?;

        let ffn_output = self.ffn.forward(&self.norm2.forward(&x)?)?;
        x.add(&ffn_output)
    }
}

/// Mini Genome Model completo
pub struct MiniGenomeModel {
    config: MiniGenomeConfig,
    token_embedding: Embedding,
    pos_embedding: SinusoidalPositionalEmbedding,
    transformer_blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    head: Linear,
    device: Device,
}

impl MiniGenomeModel {
    pub fn new(vb: VarBuilder, config: MiniGenomeConfig) -> Result<Self> {
        let device = vb.device().clone();

        // Scale token embeddings by 1/sqrt(d_model).  N(0,1) embeddings produce
        // hidden states with std ~1, which makes the attention scores too large
        // and the training trajectory highly initialization-dependent.  Scaling
        // down keeps the residual stream stable and lets the small head init
        // produce near-uniform initial logits.
        let emb_init = Init::Randn { mean: 0.0, stdev: 1.0 / (config.d_model as f64).sqrt() };
        let token_emb_weight = vb.get_with_hints((config.vocab_size, config.d_model), "weight", emb_init)?;
        let token_embedding = Embedding::new(token_emb_weight, config.d_model);

        let pos_embedding = SinusoidalPositionalEmbedding::new(config.max_seq_len, config.d_model);

        let mut transformer_blocks = Vec::new();
        for i in 0..config.n_layers {
            let block = TransformerBlock::new(vb.pp(format!("block_{}", i)), &config)?;
            transformer_blocks.push(block);
        }

        let final_norm = layer_norm(config.d_model, 1e-5, vb.pp("final_norm"))?;
        // Initialize the output head with very small random weights and zero bias.
        // With scaled token embeddings the residual stream has std ~1/sqrt(d_model),
        // so a stdev of 0.02 keeps the initial logits near a uniform distribution
        // and prevents a random class bias (e.g. always predicting C) at start-up.
        let head_weight = vb.get_with_hints((config.vocab_size, config.d_model), "weight", Init::Randn { mean: 0.0, stdev: 0.02 })?;
        let head_bias = vb.get_with_hints(config.vocab_size, "bias", Init::Const(0.0))?;
        let head = Linear::new(head_weight, Some(head_bias));

        Ok(Self { config, token_embedding, pos_embedding, transformer_blocks, final_norm, head, device })
    }

    /// Forward pass completo
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;

        let token_emb = self.token_embedding.forward(input_ids)?;

        let pos_emb = self.pos_embedding.forward(seq_len, &self.device)?;
        let pos_emb = pos_emb.unsqueeze(0)?.broadcast_as(token_emb.shape())?;

        let mut x = token_emb.add(&pos_emb)?;

        for block in &self.transformer_blocks {
            x = block.forward(&x)?;
        }

        let x = self.final_norm.forward(&x)?;
        self.head.forward(&x)
    }

    /// Calcula loss para MLM. Returns (loss, accuracy)
    ///
    /// Apenas as posicoes mascaradas (onde input_ids != labels) contribuem para
    /// o loss e para a acuracia. Isso evita que o modelo aprenda a simplesmente
    /// copiar as bases nao mascaradas e foca a previsao das bases reais do
    /// genoma que foram escondidas.
    ///
    /// Only the first 4 logits (A, C, G, T) are used for MLM; the special-token
    /// logits are masked to -inf so the model cannot waste capacity predicting
    /// [MASK], pad, [CLS] or [SEP] during genomic pre-training.  This makes the
    /// random-initialized loss ~ln(4) and removes a common source of collapse.
    pub fn compute_mlm_loss(&self, input_ids: &Tensor, labels: &Tensor, class_weights: Option<&Tensor>) -> Result<(Tensor, f32)> {
        let logits = self.forward(input_ids)?;
        let (batch, seq_len, _vocab_size) = logits.dims3()?;

        // Mask out special tokens (ids 4..vocab_size) from the output distribution.
        let logits_dna = logits.narrow(candle_core::D::Minus1, 0, 4)?;
        let logits_flat = logits_dna.reshape((batch * seq_len, 4))?;
        let labels_u32 = labels.to_dtype(DType::U32)?;
        let labels_flat = labels_u32.reshape((batch * seq_len,))?;

        // Mask: train only on positions where the input was masked (input != label).
        // Padding positions also have input == label (mask token == mask token) and
        // are therefore ignored, which is the desired behavior.  Compute this with
        // the original labels before clamping special-token ids to the valid range.
        let mask =
            input_ids.to_dtype(DType::U32)?.ne(&labels.to_dtype(DType::U32)?)?.to_dtype(DType::F32)?.reshape((batch * seq_len,))?;

        // Clamp labels to the four valid DNA bases so that padding labels (e.g.
        // the [MASK] token id) do not cause gather/argmax out-of-bounds.  Those
        // positions are excluded by `mask` so the clamped value is harmless.
        let min_label = Tensor::new(0u32, input_ids.device())?;
        let max_label = Tensor::new(3u32, input_ids.device())?;
        let labels_flat = labels_flat.broadcast_maximum(&min_label)?.broadcast_minimum(&max_label)?;

        // Per-position log-probabilities over the four DNA bases.
        let log_probs = candle_nn::ops::log_softmax(&logits_flat, candle_core::D::Minus1)?;
        let labels_flat_unsqueezed = labels_flat.unsqueeze(1)?;
        let target_log_probs = log_probs.gather(&labels_flat_unsqueezed, candle_core::D::Minus1)?;

        let smoothing = self.config.label_smoothing;
        let mut nll = if smoothing > 0.0 {
            // KL loss with uniform smoothing over the 4 active classes.
            let sum_log_probs = log_probs.sum(1)?.unsqueeze(1)?;
            let other_log_probs = (&sum_log_probs - &target_log_probs)?;
            let nll = ((&target_log_probs * (smoothing - 1.0))? - (&other_log_probs * (smoothing / 3.0))?)?;
            nll.reshape((batch * seq_len,))?
        } else {
            target_log_probs.neg()?.reshape((batch * seq_len,))?
        };

        // Optionally reweight classes (e.g. inverse-frequency) to combat class imbalance.
        // Only the first 4 weights are meaningful; special-token weights are ignored.
        if let Some(cw) = class_weights {
            let cw_dna = cw.narrow(0, 0, 4)?;
            let cw_per_token = cw_dna.index_select(&labels_flat, 0)?.reshape((batch * seq_len,))?;
            nll = nll.mul(&cw_per_token)?;
        }

        let masked_nll = (&nll * &mask)?;
        let mask_sum = mask.sum_all()?;
        let mask_sum_f = mask_sum.to_vec0::<f32>()?;
        let loss = if mask_sum_f == 0.0 { Tensor::new(0.0f32, input_ids.device())? } else { masked_nll.sum_all()?.div(&mask_sum)? };

        // Accuracy over the masked positions only, restricted to the 4 bases.
        let predictions = logits_flat.argmax(candle_core::D::Minus1)?;
        let correct = predictions.eq(&labels_flat)?.to_dtype(DType::F32)?;
        let masked_correct = (&correct * &mask)?;
        let accuracy = if mask_sum_f == 0.0 { 0.0 } else { masked_correct.sum_all()?.to_vec0::<f32>()? / mask_sum_f };

        Ok((loss, accuracy))
    }

    /// Info do modelo
    pub fn info(&self) {
        println!("{}", "=".repeat(60));
        println!("Mini Genome Model (MGM-1)");
        println!("{}", "=".repeat(60));
        println!("Configuracao:");
        println!("  d_model:      {}", self.config.d_model);
        println!("  n_heads:      {}", self.config.n_heads);
        println!("  n_layers:     {}", self.config.n_layers);
        println!("  d_ff:         {}", self.config.d_ff);
        println!("  vocab_size:   {}", self.config.vocab_size);
        println!("  max_seq_len:  {}", self.config.max_seq_len);
        println!("{}", "-".repeat(60));
        println!("Parametros totais: ~{:.2}M", self.config.num_parameters() as f64 / 1e6);
        println!("{}", "=".repeat(60));
    }
}

impl Module for MiniGenomeModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

/// Treinador simples
pub struct Trainer {
    model: MiniGenomeModel,
    optimizer: candle_nn::optim::AdamW,
    varmap: VarMap,
}

impl Trainer {
    pub fn new(model: MiniGenomeModel, varmap: VarMap, lr: f64) -> Result<Self> {
        let vars = varmap.all_vars();
        let params = candle_nn::optim::ParamsAdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01 };
        let optimizer = candle_nn::optim::AdamW::new(vars, params)?;

        Ok(Self { model, optimizer, varmap })
    }

    pub fn train_step(&mut self, input_ids: &Tensor, labels: &Tensor, class_weights: Option<&Tensor>) -> Result<(f32, f32)> {
        let (loss, _accuracy) = self.model.compute_mlm_loss(input_ids, labels, class_weights)?;

        self.optimizer.backward_step(&loss)?;

        let loss_val = loss.to_scalar::<f32>()?;

        Ok((loss_val, 0.0))
    }

    pub fn model(&self) -> &MiniGenomeModel {
        &self.model
    }

    /// Salvar modelo
    pub fn save(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_tokenizer() {
        let tokenizer = DnaTokenizer::new();
        let seq = "ACGTACGT";
        let tokens = tokenizer.encode(seq);
        assert_eq!(tokens, vec![0, 1, 2, 3, 0, 1, 2, 3]);

        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, seq);
    }

    #[test]
    fn test_tokenizer_maps_each_base_consistently() {
        let tokenizer = DnaTokenizer::new();
        // A/C/G/T must round-trip individually and match their token ids.
        assert_eq!(tokenizer.encode("A"), vec![0]);
        assert_eq!(tokenizer.encode("C"), vec![1]);
        assert_eq!(tokenizer.encode("G"), vec![2]);
        assert_eq!(tokenizer.encode("T"), vec![3]);

        assert_eq!(tokenizer.decode(&[0]), "A");
        assert_eq!(tokenizer.decode(&[1]), "C");
        assert_eq!(tokenizer.decode(&[2]), "G");
        assert_eq!(tokenizer.decode(&[3]), "T");

        // The ids also match the 2-bit genome archive encoding used by seed-node.
        assert_eq!(tokenizer.encode("ACGT"), vec![0b00, 0b01, 0b10, 0b11]);
    }

    #[test]
    fn test_model_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = MiniGenomeConfig::tiny();
        let model = MiniGenomeModel::new(vb, config).unwrap();
        model.info();
    }

    #[test]
    fn test_forward_pass() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = MiniGenomeConfig::tiny();
        let model = MiniGenomeModel::new(vb, config).unwrap();

        let mut rng = rand::thread_rng();
        let input_vec: Vec<i64> = (0..20).map(|_| rng.gen_range(0..8)).collect();
        let input_ids = Tensor::new(input_vec, &device).unwrap().reshape((2, 10)).unwrap();
        let output = model.forward(&input_ids).unwrap();

        assert_eq!(output.dims(), &[2, 10, 8]);
    }

    /// Class-weighted MLM loss accepts a 1-D weight tensor and still produces
    /// a finite, non-negative loss value.  This is a regression test for the
    /// A/T bias: it locks down the class-weight plumbing in `compute_mlm_loss`.
    #[test]
    fn test_class_weighted_mlm_loss() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = MiniGenomeConfig::tiny();
        let model = MiniGenomeModel::new(vb, config).unwrap();

        // A small batch with one masked position per row.
        let input = Tensor::new(&[[0i64, 4i64], [4i64, 1i64]], &device).unwrap();
        let labels = Tensor::new(&[[0i64, 0i64], [1i64, 1i64]], &device).unwrap();

        // Inverse-frequency-ish weights that penalize A and reward C.
        let weights = Tensor::new(&[0.5f32, 2.0f32, 1.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32], &device).unwrap();

        let (loss_cw, _) = model.compute_mlm_loss(&input, &labels, Some(&weights)).unwrap();
        let (loss_no_cw, _) = model.compute_mlm_loss(&input, &labels, None).unwrap();

        let loss_cw_f = loss_cw.to_scalar::<f32>().unwrap();
        let loss_no_cw_f = loss_no_cw.to_scalar::<f32>().unwrap();

        assert!(loss_cw_f.is_finite() && loss_no_cw_f.is_finite());
        // Class weights must change the loss value.  With scaled embeddings the
        // random model may already predict C better or worse than A, so we only
        // assert the weights have a non-trivial effect, not the direction.
        assert!(
            (loss_cw_f - loss_no_cw_f).abs() > 1e-4,
            "Class-weighted loss {} is too close to unweighted {}",
            loss_cw_f,
            loss_no_cw_f
        );
    }
}
