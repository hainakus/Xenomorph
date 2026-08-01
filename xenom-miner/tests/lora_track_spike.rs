//! End-to-end LoRA Track spike.
//!
//! Starts a mock orchestrator that holds a tiny DNABERT-2 model and exposes the
//! new `GetTrainingArtifact`, `AttestedForward`, and `SubmitLoRAUpdate` RPCs.
//! A `LoraOnlyTrainer` then runs the full loop and trains a LoRA adapter on the
//! LM head without loading the base transformer.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use borsh::{to_vec, BorshDeserialize};
use candle_core::{DType, Device, Tensor};
use futures::{SinkExt, StreamExt};
use model_crypto::artifact_sign::ArtifactSigner;
use model_crypto::key_hierarchy::ModelKeyHierarchy;
use model_crypto::session;
use secp256k1::PublicKey;
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer};
use tokio::net::TcpListener;
use tokio::time::timeout;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

use xenom_miner::dnabert2::DnaBert2ForMaskedLM;
use xenom_miner::lora::LoraConfig;
use xenom_miner::model::DnaBert2Config;
use xenom_miner::rpc::messages::{ArtifactType, AttestedForwardResponse, RpcEnvelope, RpcRequest, RpcResponse, TrainingArtifact};
use xenom_miner::rpc::XenomRpcClient;
use xenom_miner::trainer::lora_only_trainer::LoraOnlyTrainer;

const TEST_TIMEOUT: Duration = Duration::from_secs(60);

fn build_tiny_config() -> DnaBert2Config {
    DnaBert2Config {
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
    std::fs::read(&path).unwrap()
}

fn build_test_tokenizer_bytes() -> Vec<u8> {
    let mut vocab = tokenizers::models::bpe::Vocab::new();
    vocab.insert("A".to_string(), 0);
    vocab.insert("T".to_string(), 1);
    vocab.insert("C".to_string(), 2);
    vocab.insert("G".to_string(), 3);
    vocab.insert("<mask>".to_string(), 4);
    vocab.insert("<pad>".to_string(), 5);

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

    let bpe = BPE::new(vocab, vec![]);
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true), AddedToken::from("<pad>", true)]);
    serde_json::to_vec(&tokenizer).expect("failed to serialize test tokenizer")
}

fn build_tiny_artifact(weights: &[u8], config: &DnaBert2Config, tokenizer: &[u8], miner_public_key: &[u8; 33]) -> TrainingArtifact {
    let device = Device::Cpu;
    let loaded = candle_core::safetensors::load_buffer(weights, &device).unwrap();

    let artifact_tensors: Vec<(String, &Tensor)> = vec![
        ("lm_head.transform.dense.weight".to_string(), loaded.get("lm_head.transform.dense.weight").unwrap()),
        ("lm_head.transform.dense.bias".to_string(), loaded.get("lm_head.transform.dense.bias").unwrap()),
        ("lm_head.transform.layer_norm.weight".to_string(), loaded.get("lm_head.transform.layer_norm.weight").unwrap()),
        ("lm_head.transform.layer_norm.bias".to_string(), loaded.get("lm_head.transform.layer_norm.bias").unwrap()),
        ("lm_head.bias".to_string(), loaded.get("lm_head.bias").unwrap()),
        ("model.embeddings.word_embeddings.weight".to_string(), loaded.get("model.embeddings.word_embeddings.weight").unwrap()),
    ];

    let artifact = ::safetensors::tensor::serialize(artifact_tensors, &None).unwrap();
    let artifact_hash = *blake3::hash(&artifact).as_bytes();

    // Sign the artifact with a dummy model hierarchy so the trainer's
    // signature verification passes.
    let hierarchy = ModelKeyHierarchy::random("xeno/mgm-1", 1);
    let auth_key = hierarchy.auth_key().unwrap();
    let signer = ArtifactSigner::from_auth_key(&auth_key).unwrap();
    let auth_public_key = signer.public_key();
    let signature = signer.sign(&artifact_hash, Some(&[2u8; 32])).unwrap();

    // Encrypt the artifact to the miner's public key using ECDH.
    let miner_pk = PublicKey::from_slice(miner_public_key).unwrap();
    let (ephemeral_secret, ephemeral_public_key) = session::generate_ephemeral_keypair();
    let session_nonce = [0u8; 12];
    let shared_secret = session::orchestrator_shared_secret(&ephemeral_secret, &miner_pk);
    let session_key = session::derive_session_key(&shared_secret, &session_nonce).unwrap();
    let encrypted_artifact = session_key.encrypt(&artifact).unwrap();

    TrainingArtifact {
        model_id: "xeno/mgm-1".to_string(),
        base_checkpoint: [1u8; 32],
        base_hash: [2u8; 32],
        config: serde_json::to_vec(config).unwrap(),
        tokenizer: tokenizer.to_vec(),
        artifact: encrypted_artifact,
        artifact_type: ArtifactType::LoRA,
        artifact_hash,
        encrypted: true,
        recipient_key_fingerprint: [0u8; 32],
        signature: signature.signature,
        ephemeral_public_key: ephemeral_public_key.serialize(),
        session_nonce,
        auth_public_key,
    }
}

fn compute_masked_loss(logits: &Tensor, labels: &Tensor, mask: &Tensor, device: &Device) -> f64 {
    let dims = logits.dims();
    let (b, s, v) = (dims[0], dims[1], dims[2]);
    let logits_flat = logits.reshape((b * s, v)).unwrap();
    let labels_flat = labels.reshape((b * s,)).unwrap();
    let mask_flat = mask.flatten_all().unwrap();
    let mask_vec = mask_flat.to_vec1::<u8>().unwrap();
    let positions: Vec<u32> = mask_vec.iter().enumerate().filter(|(_, &m)| m != 0).map(|(i, _)| i as u32).collect();
    let positions_t = Tensor::new(positions.as_slice(), device).unwrap();
    let masked_logits = logits_flat.index_select(&positions_t, 0).unwrap();
    let labels_vec = labels_flat.to_vec1::<u32>().unwrap();
    let masked_labels: Vec<u32> = positions.iter().map(|&i| labels_vec[i as usize]).collect();
    let masked_labels = Tensor::new(masked_labels.as_slice(), device).unwrap();
    let masked_logits_f32 = masked_logits.to_dtype(DType::F32).unwrap();
    candle_nn::loss::cross_entropy(&masked_logits_f32, &masked_labels).unwrap().to_vec0::<f32>().unwrap() as f64
}

async fn start_mock_orchestrator(weights: Vec<u8>, config: DnaBert2Config, tokenizer: Vec<u8>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let config = Arc::new(config);
    let tokenizer = Arc::new(tokenizer);
    let weights = Arc::new(weights);

    tokio::spawn(async move {
        let device = Device::Cpu;
        let model = DnaBert2ForMaskedLM::load((*config).clone(), (*weights).clone(), DType::F32, &device, None).unwrap();

        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        while let Some(Ok(msg)) = ws.next().await {
            if let Message::Binary(bytes) = msg {
                let envelope: RpcEnvelope = match RpcEnvelope::try_from_slice(&bytes) {
                    Ok(env) => env,
                    Err(_) => {
                        let _ = ws.send(Message::Binary(to_vec(&RpcResponse::Error("bad request".to_string())).unwrap())).await;
                        continue;
                    }
                };

                let response = match envelope.payload {
                    RpcRequest::GetTrainingArtifact(req) => {
                        let artifact = build_tiny_artifact(&weights, &config, &tokenizer, &req.miner_public_key);
                        RpcResponse::TrainingArtifact(artifact)
                    }
                    RpcRequest::AttestedForward(req) => {
                        let batch_size = req.input_ids.len();
                        let seq_len = req.input_ids[0].len();
                        let input_ids = Tensor::from_vec(
                            req.input_ids.iter().flatten().copied().collect::<Vec<_>>(),
                            (batch_size, seq_len),
                            &device,
                        )
                        .unwrap();
                        let attention_mask = Tensor::from_vec(
                            req.attention_mask.iter().flatten().copied().collect::<Vec<_>>(),
                            (batch_size, seq_len),
                            &device,
                        )
                        .unwrap();

                        let hidden_states = model.encode(&input_ids, None, Some(&attention_mask)).unwrap();
                        let logits = model.forward_from_hidden_states(&hidden_states).unwrap();

                        let labels =
                            Tensor::from_vec(req.labels.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &device)
                                .unwrap();
                        let mask =
                            Tensor::from_vec(req.mask.iter().flatten().copied().collect::<Vec<_>>(), (batch_size, seq_len), &device)
                                .unwrap();

                        let loss_scalar = compute_masked_loss(&logits, &labels, &mask, &device);

                        let hidden_states_f32 = hidden_states.to_dtype(DType::F32).unwrap();
                        let flat = hidden_states_f32.reshape((batch_size * seq_len * config.hidden_size,)).unwrap();
                        let hidden_vec = flat.to_vec1::<f32>().unwrap();
                        let hidden_bytes: Vec<u8> = hidden_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let hidden_states_hash = *blake3::hash(&hidden_bytes).as_bytes();

                        // Sign the response with a dummy model hierarchy.
                        let hierarchy = ModelKeyHierarchy::random("xeno/mgm-1", 1);
                        let auth_key = hierarchy.auth_key().unwrap();
                        let signer = ArtifactSigner::from_auth_key(&auth_key).unwrap();
                        let auth_public_key = signer.public_key();
                        let mut hasher = blake3::Hasher::new();
                        hasher.update(b"xenom-attested-forward-v1");
                        hasher.update(&hidden_states_hash);
                        hasher.update(&req.base_checkpoint);
                        hasher.update(&loss_scalar.to_le_bytes());
                        let message_hash: [u8; 32] = *hasher.finalize().as_bytes();
                        let signature = signer.sign(&message_hash, None).unwrap();

                        // Encrypt the hidden states to the miner's public key.
                        let miner_pk = PublicKey::from_slice(&req.miner_public_key).unwrap();
                        let (ephemeral_secret, ephemeral_public_key) = session::generate_ephemeral_keypair();
                        let session_nonce = [0u8; 12];
                        let shared_secret = session::orchestrator_shared_secret(&ephemeral_secret, &miner_pk);
                        let session_key = session::derive_session_key(&shared_secret, &session_nonce).unwrap();
                        let encrypted_hidden_states = session_key.encrypt(&hidden_bytes).unwrap();

                        RpcResponse::AttestedForward(AttestedForwardResponse {
                            hidden_states_hash,
                            hidden_states: encrypted_hidden_states,
                            loss: loss_scalar,
                            token_count: (batch_size * seq_len) as u32,
                            signature: signature.signature,
                            ephemeral_public_key: ephemeral_public_key.serialize(),
                            session_nonce,
                            auth_public_key,
                        })
                    }
                    RpcRequest::SubmitLoRAUpdate(_req) => RpcResponse::LoRAUpdateAck { new_checkpoint: Some([3u8; 32]) },
                    _ => RpcResponse::Error("unexpected request in spike".to_string()),
                };

                let payload = to_vec(&response).unwrap();
                if ws.send(Message::Binary(payload)).await.is_err() {
                    break;
                }
            }
        }
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    port
}

#[tokio::test]
async fn test_lora_track_spike() {
    let config = build_tiny_config();
    let weights = build_tiny_safetensors();
    let tokenizer = build_test_tokenizer_bytes();

    let port = start_mock_orchestrator(weights, config.clone(), tokenizer).await;
    let url = format!("ws://127.0.0.1:{}", port);

    let mut client = XenomRpcClient::new(url);
    timeout(TEST_TIMEOUT, client.connect()).await.expect("connect timed out").expect("connect failed");

    let batch_size = 2;
    let seq_len = 4;
    let input_ids: Vec<Vec<u32>> = (0..batch_size)
        .map(|b| (0..seq_len).map(|i| ((b * seq_len + i) % (config.vocab_size as usize - 1) + 1) as u32).collect())
        .collect();
    let attention_mask: Vec<Vec<u32>> = vec![vec![1u32; seq_len]; batch_size];
    let labels = input_ids.clone();
    let mask: Vec<Vec<u8>> = vec![vec![1u8; seq_len]; batch_size];

    let lora_config =
        LoraConfig { rank: 2, alpha: 4.0, dropout: 0.0, target_modules: ["dense".to_string()].iter().cloned().collect() };
    let mut trainer = LoraOnlyTrainer::new(client, 1e-2, 4, lora_config);
    let (loss, _delta) = timeout(TEST_TIMEOUT, trainer.train_round("xeno/mgm-1", [1u8; 32], input_ids, attention_mask, labels, mask))
        .await
        .expect("train round timed out")
        .expect("train round failed");

    assert!(loss.is_finite(), "LoRA-only training produced non-finite loss: {}", loss);
    assert!(loss > 0.0, "LoRA-only training loss should be positive before convergence");
}
