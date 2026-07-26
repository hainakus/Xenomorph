pub mod cert;
pub mod client;
pub mod protocol;
pub mod server;

pub use async_trait::async_trait;
pub use client::{content_hash, CheckpointTransferClient};
pub use protocol::{CheckpointFileRequest, CheckpointFileResponseHeader, CheckpointFileType, ResponseStatus};
pub use server::{CheckpointTransferServer, FileProvider};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::sync::Arc;
    use std::time::Instant;

    use async_trait::async_trait;

    use crate::client::content_hash;
    use crate::protocol::{CheckpointFileRequest, CheckpointFileType};
    use crate::{CheckpointTransferClient, CheckpointTransferServer, FileProvider};

    type Key = (String, [u8; 32], CheckpointFileType);

    struct InMemoryProvider {
        data: HashMap<Key, Vec<u8>>,
    }

    #[async_trait]
    impl FileProvider for InMemoryProvider {
        async fn get_file(&self, request: &CheckpointFileRequest) -> anyhow::Result<Option<(Vec<u8>, [u8; 32])>> {
            let key = (request.model_id.clone(), request.weights_hash, request.file_type);
            Ok(self.data.get(&key).map(|bytes| (bytes.clone(), content_hash(bytes))))
        }
    }

    struct BadHashProvider;

    #[async_trait]
    impl FileProvider for BadHashProvider {
        async fn get_file(&self, _request: &CheckpointFileRequest) -> anyhow::Result<Option<(Vec<u8>, [u8; 32])>> {
            let bytes = b"tampered".to_vec();
            let mut wrong_hash = [0u8; 32];
            wrong_hash[0] = 0xff;
            Ok(Some((bytes, wrong_hash)))
        }
    }

    fn test_provider(data: HashMap<Key, Vec<u8>>) -> Arc<dyn FileProvider> {
        Arc::new(InMemoryProvider { data })
    }

    fn localhost() -> SocketAddr {
        (Ipv4Addr::new(127, 0, 0, 1), 0).into()
    }

    #[tokio::test]
    async fn test_roundtrip_1kb() {
        let mut data = HashMap::new();
        data.insert(("multimolecule/dnabert2".to_string(), [1u8; 32], CheckpointFileType::Weights), vec![0u8; 1024]);

        let server = CheckpointTransferServer::new(localhost(), test_provider(data)).await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let client = CheckpointTransferClient::new(localhost(), "localhost").unwrap();

        let request = CheckpointFileRequest {
            model_id: "multimolecule/dnabert2".to_string(),
            weights_hash: [1u8; 32],
            file_type: CheckpointFileType::Weights,
        };
        let payload = client.get_file(server_addr, &request).await.unwrap();

        assert_eq!(payload.len(), 1024);
    }

    #[tokio::test]
    async fn test_roundtrip_4mb() {
        let mut data = HashMap::new();
        data.insert(("multimolecule/dnabert2".to_string(), [2u8; 32], CheckpointFileType::Weights), vec![0u8; 4 * 1024 * 1024]);

        let server = CheckpointTransferServer::new(localhost(), test_provider(data)).await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let client = CheckpointTransferClient::new(localhost(), "localhost").unwrap();

        let request = CheckpointFileRequest {
            model_id: "multimolecule/dnabert2".to_string(),
            weights_hash: [2u8; 32],
            file_type: CheckpointFileType::Weights,
        };

        let start = Instant::now();
        let payload = client.get_file(server_addr, &request).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(payload.len(), 4 * 1024 * 1024);
        // Localhost 4 MB should complete quickly; allow 300 ms to account for
        // hash verification overhead in debug builds.
        assert!(elapsed.as_millis() < 300, "4 MB transfer took {} ms", elapsed.as_millis());
    }

    #[tokio::test]
    async fn test_not_found() {
        let server = CheckpointTransferServer::new(localhost(), test_provider(HashMap::new())).await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let client = CheckpointTransferClient::new(localhost(), "localhost").unwrap();

        let request = CheckpointFileRequest {
            model_id: "multimolecule/dnabert2".to_string(),
            weights_hash: [99u8; 32],
            file_type: CheckpointFileType::Config,
        };

        let err = client.get_file(server_addr, &request).await.unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_all_file_types_roundtrip() {
        let mut data = HashMap::new();
        data.insert(("model".to_string(), [3u8; 32], CheckpointFileType::Config), b"{}".to_vec());
        data.insert(("model".to_string(), [3u8; 32], CheckpointFileType::Tokenizer), b"[]".to_vec());
        data.insert(("model".to_string(), [3u8; 32], CheckpointFileType::Weights), vec![7u8; 64]);
        data.insert(("model".to_string(), [3u8; 32], CheckpointFileType::Adapter), vec![8u8; 32]);

        let server = CheckpointTransferServer::new(localhost(), test_provider(data)).await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let client = CheckpointTransferClient::new(localhost(), "localhost").unwrap();

        for file_type in
            [CheckpointFileType::Config, CheckpointFileType::Tokenizer, CheckpointFileType::Weights, CheckpointFileType::Adapter]
        {
            let request = CheckpointFileRequest { model_id: "model".to_string(), weights_hash: [3u8; 32], file_type };
            let payload = client.get_file(server_addr, &request).await.unwrap();
            assert!(!payload.is_empty(), "{:?} payload was empty", file_type);
        }
    }

    #[tokio::test]
    async fn test_hash_mismatch_is_rejected() {
        let server = CheckpointTransferServer::new(localhost(), Arc::new(BadHashProvider)).await.unwrap();
        let server_addr = server.local_addr().unwrap();
        let client = CheckpointTransferClient::new(localhost(), "localhost").unwrap();

        let request =
            CheckpointFileRequest { model_id: "model".to_string(), weights_hash: [1u8; 32], file_type: CheckpointFileType::Weights };

        let err = client.get_file(server_addr, &request).await.unwrap_err();
        assert!(err.to_string().contains("hash mismatch"));
    }

    #[tokio::test]
    async fn test_semaphore_caps_concurrent_transfers() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Mutex;

        struct SlowProvider {
            active: AtomicUsize,
            max_active: AtomicUsize,
            gate: Mutex<()>,
        }

        #[async_trait]
        impl FileProvider for SlowProvider {
            async fn get_file(&self, _request: &CheckpointFileRequest) -> anyhow::Result<Option<(Vec<u8>, [u8; 32])>> {
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                let max = self.max_active.load(Ordering::SeqCst);
                if active > max {
                    self.max_active.store(active, Ordering::SeqCst);
                }
                // Hold the permit while two requests are in flight.
                let _g = self.gate.lock().await;
                self.active.fetch_sub(1, Ordering::SeqCst);
                Ok(Some((b"x".to_vec(), content_hash(b"x"))))
            }
        }

        let provider = Arc::new(SlowProvider { active: AtomicUsize::new(0), max_active: AtomicUsize::new(0), gate: Mutex::new(()) });

        // Max 2 concurrent transfers.
        let server = CheckpointTransferServer::with_max_transfers(localhost(), provider.clone(), 2).await.unwrap();
        let server_addr = server.local_addr().unwrap();

        // Fire 5 requests at once. With the gate locked until the first completes,
        // only 2 should enter `get_file` at a time.
        let requests: Vec<_> = (0..5)
            .map(|i| CheckpointFileRequest {
                model_id: "model".to_string(),
                weights_hash: [i as u8; 32],
                file_type: CheckpointFileType::Weights,
            })
            .collect();

        let mut handles = Vec::new();
        for req in requests {
            let client = CheckpointTransferClient::new(localhost(), "localhost").unwrap();
            handles.push(tokio::spawn(async move {
                client.get_file(server_addr, &req).await.unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // The gate mutex serializes the slow section, so the most requests that
        // can be inside `get_file` simultaneously is the configured limit.
        assert!(provider.max_active.load(Ordering::SeqCst) <= 2, "max concurrent transfers exceeded limit");
    }
}
