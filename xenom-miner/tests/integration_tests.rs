use std::time::Duration;

use borsh::{to_vec, BorshDeserialize};
use futures::{SinkExt, StreamExt};
use kaspa_consensus_core::network::NetworkType;
use tokio::net::TcpListener;
use tokio::time::timeout;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

use xenom_miner::block::BlockBuilder;
use xenom_miner::model_cache::ModelCache;
use xenom_miner::model_client::fetch_model_checkpoint;
use xenom_miner::prover::{PublicInputs, ZkProver};
use xenom_miner::rpc::messages::{
    BlockHeader, ModelCheckpointInfo, ModelCheckpointInfoV2, ModelCheckpointV2, RpcEnvelope, RpcRequest, RpcResponse, TrainingBatch,
    TrainingBlock, TrainingProof,
};
use xenom_miner::rpc::XenomRpcClient;
use xenom_miner::trainer::{MockTrainer, Trainer};
use xenom_miner::wallet::WalletManager;

/// Start a minimal mock Xenomorph node that speaks Borsh over WebSocket.
async fn start_mock_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        // Consistent payload/hash so `fetch_model_checkpoint` hash verification passes.
        let weights = vec![0u8; 64];
        let combined = *blake3::hash(&weights).as_bytes();
        let base_hash = [2u8; 32];

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
                    RpcRequest::GetTrainingBatch { model_id } => RpcResponse::TrainingBatch(Some(TrainingBatch {
                        batch_id: 42,
                        model_id,
                        base_checkpoint: combined,
                        data_indices: vec![0, 1, 2, 3],
                        target_improvement: 0.01,
                        learning_rate: 0.01,
                    })),
                    RpcRequest::GetModelCheckpoint { model_id } => {
                        RpcResponse::ModelCheckpoint(xenom_miner::rpc::messages::ModelCheckpoint {
                            model_id,
                            base_checkpoint: combined,
                            config: b"{}".to_vec(),
                            tokenizer: b"[]".to_vec(),
                            weights: weights.clone(),
                            encrypted: false,
                        })
                    }
                    RpcRequest::SubmitBlock(_) => RpcResponse::BlockHash([7u8; 32]),
                    RpcRequest::Heartbeat => RpcResponse::Pong,
                    RpcRequest::GetBalance { .. } => RpcResponse::Balance(0),
                    RpcRequest::GetDifficulty => RpcResponse::Difficulty([0xff; 32]),
                    RpcRequest::GetGenomeTrainingBatch(_) => RpcResponse::Error("genome batch not supported in mock".to_string()),
                    RpcRequest::GetModelCheckpointInfo(req) => {
                        RpcResponse::ModelCheckpointInfo(ModelCheckpointInfo { model_id: req.model_id, base_checkpoint: combined })
                    }
                    RpcRequest::GetModelCheckpointInfoV2(req) => RpcResponse::ModelCheckpointInfoV2(ModelCheckpointInfoV2 {
                        model_id: req.model_id,
                        base_checkpoint: combined,
                        base_hash,
                    }),
                    RpcRequest::GetModelCheckpointV2(req) => RpcResponse::ModelCheckpointV2(ModelCheckpointV2 {
                        model_id: req.model_id,
                        base_checkpoint: combined,
                        base_hash,
                        config: b"{}".to_vec(),
                        tokenizer: b"[]".to_vec(),
                        weights: weights.clone(),
                        encrypted: false,
                        is_adapter: false,
                    }),
                    RpcRequest::SubmitGradients(_) => RpcResponse::GradientAck { new_checkpoint: None },
                    RpcRequest::GetCheckpointPeers(_req) => RpcResponse::CheckpointPeers(Vec::new()),
                    RpcRequest::GetTrainingArtifact(_req) => {
                        RpcResponse::Error("GetTrainingArtifact not supported in mock".to_string())
                    }
                    RpcRequest::AttestedForward(_req) => RpcResponse::Error("AttestedForward not supported in mock".to_string()),
                    RpcRequest::SubmitLoRAUpdate(_req) => RpcResponse::LoRAUpdateAck { new_checkpoint: None },
                };

                let payload = to_vec(&response).unwrap();
                if ws.send(Message::Binary(payload)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Give the server a moment to start listening.
    tokio::time::sleep(Duration::from_millis(100)).await;
    port
}

#[tokio::test]
async fn test_rpc_client_against_mock_server() {
    let port = start_mock_server().await;
    let url = format!("ws://127.0.0.1:{}", port);

    let mut client = XenomRpcClient::new(url);
    timeout(TEST_TIMEOUT, client.connect()).await.expect("client connect timed out").expect("client connect failed");

    let batch = client.get_training_batch("dnabert2").await.expect("get_training_batch failed").expect("server returned no batch");
    assert_eq!(batch.batch_id, 42);
    assert_eq!(batch.model_id, "dnabert2");

    let block = TrainingBlock {
        header: BlockHeader {
            prev_block_hash: [0u8; 32],
            block_number: 1,
            timestamp: 0,
            merkle_root: [1u8; 32],
            difficulty: [2u8; 32],
            nonce: 0,
        },
        model_id: "dnabert2".to_string(),
        training_proof: TrainingProof {
            base_checkpoint: [3u8; 32],
            loss_before: 2.45,
            loss_after: 2.41,
            gradients_commitment: [4u8; 32],
            zk_proof: vec![0u8; 32],
            batch_indices: vec![0, 1, 2],
            compute_time_ms: 100,
        },
        miner_address: "xnom:test".to_string(),
        timestamp: 0,
        signature: [0u8; 64],
    };

    let hash = timeout(TEST_TIMEOUT, client.submit_block(block)).await.expect("submit_block timed out").expect("submit_block failed");
    assert_eq!(hash, [7u8; 32]);
}

#[tokio::test]
async fn test_end_to_end_mining_pipeline() {
    let tmp = tempfile::tempdir().unwrap();
    let wallet = WalletManager::create_new(tmp.path(), "password", NetworkType::Devnet).unwrap();

    let batch = TrainingBatch {
        batch_id: 1,
        model_id: "dnabert2".to_string(),
        base_checkpoint: [0u8; 32],
        data_indices: vec![0, 1, 2, 3],
        target_improvement: 0.01,
        learning_rate: 0.01,
    };

    let trainer = MockTrainer::new();
    let result = trainer.train(&batch).unwrap();
    assert!(result.loss_after < result.loss_before);

    let prover = ZkProver::new();
    let public_inputs = PublicInputs {
        model_id: batch.model_id.clone(),
        batch_id: batch.batch_id,
        loss_before: result.loss_before,
        loss_after: result.loss_after,
        gradients_commitment: result.gradients_commitment,
        base_checkpoint: result.base_checkpoint,
    };
    let proof = prover.generate_proof(&result, &public_inputs).unwrap();
    assert!(prover.verify_proof(&proof, &result, &public_inputs));

    let mut builder = BlockBuilder::new(wallet.address().to_string());
    let mut block = builder.build_block("dnabert2", &result, proof, [0u8; 32]).unwrap();
    wallet.sign_block(&mut block).unwrap();
    assert!(wallet.verify_signature(&block).unwrap());
}

#[tokio::test]
async fn test_fetch_model_checkpoint_falls_back_to_websocket() {
    let tmp = tempfile::tempdir().unwrap();
    let port = start_mock_server().await;
    let url = format!("ws://127.0.0.1:{}", port);

    let mut client = XenomRpcClient::new(url);
    timeout(TEST_TIMEOUT, client.connect()).await.expect("connect timed out").expect("connect failed");

    let cache = ModelCache::new(tmp.path().join("models"));
    // The miner always fetches over WebSocket and verifies the downloaded weights hash.
    let bundle = timeout(TEST_TIMEOUT, fetch_model_checkpoint(&mut client, "dnabert2", &cache))
        .await
        .expect("fetch_model_checkpoint timed out")
        .expect("fetch_model_checkpoint failed");

    assert_eq!(bundle.model_id, "dnabert2");
    assert!(!bundle.config.is_empty());
    assert!(!bundle.tokenizer.is_empty());
    assert!(!bundle.weights.is_empty());
}
