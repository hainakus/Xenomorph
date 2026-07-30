#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};
    use tokenizers::models::bpe::Vocab;
    use tokenizers::tokenizer::AddedToken;

    use crate::model::DnaBert2Config;
    use crate::rpc::messages::{GenomeSlice, GenomeTrainingBatch, GenomeTrainingBatchMsg, TrainingBatch};
    use crate::tokenizer::DnaTokenizer;
    use crate::trainer::{DeviceType, GpuBackend, GpuTrainer, Trainer};

    fn build_tiny_tokenizer() -> DnaTokenizer {
        let mut vocab: Vocab = Vocab::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("A".to_string(), 1);
        vocab.insert("T".to_string(), 2);
        vocab.insert("C".to_string(), 3);
        vocab.insert("G".to_string(), 4);
        vocab.insert("<mask>".to_string(), 5);

        // Add 2-mers so the tiny tokenizer is a realistic BPE/k-mer vocabulary.
        let bases = ['A', 'T', 'C', 'G'];
        let mut id = 6u32;
        for a in bases {
            for b in bases {
                let mut kmer = String::with_capacity(2);
                kmer.push(a);
                kmer.push(b);
                vocab.insert(kmer, id);
                id += 1;
            }
        }

        let bpe = tokenizers::models::bpe::BPE::new(vocab, vec![]);
        let mut tokenizer = tokenizers::Tokenizer::new(bpe);
        tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true), AddedToken::from("<pad>", true)]);

        let bytes = serde_json::to_vec(&tokenizer).unwrap();
        DnaTokenizer::from_bytes(&bytes).unwrap()
    }

    fn insert_weight(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize], device: &Device) {
        let n = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() + 0.001).collect();
        let t = Tensor::from_vec(data, shape, device).unwrap();
        map.insert(name.to_string(), t);
    }

    fn build_tiny_safetensors() -> (DnaBert2Config, Vec<u8>) {
        let device = Device::Cpu;
        // 22 = <pad>, A, T, C, G, <mask> (6) + 16 DNA 2-mers.
        let config = DnaBert2Config {
            vocab_size: 22,
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
            mask_token_id: 5,
            bos_token_id: 1,
            eos_token_id: 2,
            num_labels: None,
        };

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        insert_weight(&mut tensors, "model.embeddings.word_embeddings.weight", &[config.vocab_size, config.hidden_size], &device);
        insert_weight(
            &mut tensors,
            "model.embeddings.token_type_embeddings.weight",
            &[config.type_vocab_size, config.hidden_size],
            &device,
        );
        insert_weight(&mut tensors, "model.embeddings.layer_norm.weight", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.bias", &[config.hidden_size], &device);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.encoder.layer.{}", i);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.query.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.query.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.key.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.key.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.value.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.value.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.output.dense.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.output.dense.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.weight", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.bias", prefix), &[config.hidden_size], &device);

            insert_weight(
                &mut tensors,
                &format!("{}.mlp.up_proj.weight", prefix),
                &[config.intermediate_size * 2, config.hidden_size],
                &device,
            );
            insert_weight(
                &mut tensors,
                &format!("{}.mlp.down_proj.weight", prefix),
                &[config.hidden_size, config.intermediate_size],
                &device,
            );
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
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let weights = std::fs::read(&path).unwrap();
        (config, weights)
    }

    fn dummy_batch() -> TrainingBatch {
        TrainingBatch {
            batch_id: 1,
            model_id: "dnabert2".to_string(),
            base_checkpoint: [0u8; 32],
            data_indices: vec![0, 1, 2, 3],
            target_improvement: 0.01,
            learning_rate: 0.01,
        }
    }

    fn dummy_genome_msg() -> GenomeTrainingBatchMsg {
        GenomeTrainingBatchMsg {
            batch: GenomeTrainingBatch {
                batch_id: 1,
                model_id: "dnabert2".to_string(),
                genome_merkle_root: [1u8; 32],
                data_indices: vec![
                    GenomeSlice { chunk_idx: 0, start_base: 0, length: 4 },
                    GenomeSlice { chunk_idx: 1, start_base: 0, length: 4 },
                ],
                mask_ratio: 0.15,
                seq_length: 8,
            },
            sequences: vec!["ATCG".to_string(), "GCTA".to_string()],
            base_checkpoint: [1u8; 32],
        }
    }

    #[test]
    fn test_gpu_trainer_cpu_fallback() {
        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let trainer = GpuTrainer::new(config, weights, tokenizer, GpuBackend::Auto, 0, false, 2, None).unwrap();

        let info = trainer.device_info();
        assert_eq!(info.device_type, DeviceType::Cpu);
        assert!(info.name.contains("CPU"));

        let result = trainer.train(&dummy_batch()).unwrap();
        assert_eq!(result.model_id, "dnabert2");
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
        assert!(result.loss_after <= result.loss_before);
    }

    #[test]
    fn test_gpu_trainer_genome_fallback() {
        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let trainer = GpuTrainer::new(config, weights, tokenizer, GpuBackend::Auto, 0, false, 2, None).unwrap();

        let result = trainer.train_genome(&dummy_genome_msg()).unwrap();
        assert_eq!(result.model_id, "dnabert2");
        // Reverse-complement augmentation doubles the rows (forward + RC for each source index).
        assert_eq!(result.batch_indices, vec![0, 0, 1, 1]);
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
        assert!(result.loss_after <= result.loss_before);
    }

    #[test]
    fn test_multi_gpu_trainer_state_persists() {
        use crate::trainer::{MultiGpuConfig, MultiGpuTrainer};
        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let gpu_config = MultiGpuConfig {
            gpus: vec![0],
            micro_batch_size: 2,
            gradient_accumulation_steps: 1,
            use_mixed_precision: false,
            use_gradient_checkpointing: false,
            zero_optimization: 0,
            gradient_top_k_ratio: 1.0,
            lora_config: None,
            max_seq_len: 512,
        };
        let trainer =
            MultiGpuTrainer::new("dnabert2".to_string(), config, weights, tokenizer, gpu_config, GpuBackend::Auto, 2).unwrap();

        let result1 = trainer.train(&dummy_batch()).unwrap();
        let result2 = trainer.train(&dummy_batch()).unwrap();
        // Each batch trains from the same base checkpoint, then the replica is
        // restored to that base. The second batch should therefore start from
        // the same loss as the first batch (not from the improved state).
        // A small tolerance is allowed for floating-point accumulation differences
        // between micro-batch loss aggregation and a full-batch loss pass.
        assert!(
            (result2.loss_before - result1.loss_before).abs() <= 1e-1,
            "MultiGpuTrainer did not reset to base: {} != {}",
            result2.loss_before,
            result1.loss_before
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_available() {
        // This test only runs when the cuda feature is enabled. It will still
        // pass on machines without a GPU because backend_available returns false.
        let _ = GpuTrainer::backend_available(GpuBackend::Cuda, 0);
    }
}
