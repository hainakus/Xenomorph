pub mod cert;
pub mod client;
pub mod protocol;
pub mod server;

pub use client::CheckpointTransferClient;
pub use server::{CheckpointTransferServer, FileProvider};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::sync::Arc;
    use std::time::Instant;

    use crate::protocol::{CheckpointFileRequest, CheckpointFileType};
    use crate::{CheckpointTransferClient, CheckpointTransferServer, FileProvider};

    fn test_provider(data: HashMap<(String, [u8; 32], CheckpointFileType), Vec<u8>>) -> FileProvider {
        let data = Arc::new(data);
        Arc::new(move |req| {
            let key = (req.model_id.clone(), req.weights_hash, req.file_type);
            Ok(data.get(&key).cloned())
        })
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
        // Localhost 4 MB should comfortably complete in under 200 ms.
        assert!(elapsed.as_millis() < 200, "4 MB transfer took {} ms", elapsed.as_millis());
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
}
