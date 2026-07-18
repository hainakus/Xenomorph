use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};
use tracing::{info, instrument};

use crate::model::manager::ModelManager;
use crate::rpc::client::XenomorphRpcClient;
use crate::serving::proof::ProofGenerator;

pub mod xenom {
    pub mod inference {
        tonic::include_proto!("xenom.inference");
    }
}

use xenom::inference::inference_server::{Inference, InferenceServer};
use xenom::inference::*;

pub struct InferenceService {
    model_manager: Arc<ModelManager>,
    xenomorph_client: Arc<XenomorphRpcClient>,
    proof_generator: ProofGenerator,
}

impl InferenceService {
    pub fn new(model_manager: Arc<ModelManager>, xenomorph_client: Arc<XenomorphRpcClient>) -> Self {
        Self { model_manager, xenomorph_client, proof_generator: ProofGenerator::new() }
    }
}

#[tonic::async_trait]
impl Inference for InferenceService {
    #[instrument(skip(self, request))]
    async fn predict(&self, request: Request<PredictRequest>) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        info!("Predict request for model: {}", req.model_id);

        // Load model
        let model_info = self
            .model_manager
            .load_model(&req.model_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to load model: {}", e)))?;

        // Update last used
        let _ = self.model_manager.update_last_used(&req.model_id).await;

        // Simulate inference (in production, would use actual model)
        let output_data = simulate_inference(&req.input_data, &req.model_id);
        let confidence = 0.95 + (rand::random::<f32>() * 0.05);

        // Generate proof of service
        let proof_of_service = self.proof_generator.generate_proof(&req.model_id, &req.query_id, start.elapsed().as_millis() as u64);

        let latency_ms = start.elapsed().as_millis() as u64;

        let response = PredictResponse {
            output_data,
            confidence,
            proof_of_service,
            model_version: model_info.version.to_string(),
            latency_ms,
            seed_node_id: self.model_manager.node_id().to_string(),
            signature: vec![],
        };

        info!("Predict completed in {}ms", latency_ms);
        Ok(Response::new(response))
    }

    #[instrument(skip(self, request))]
    async fn embed(&self, request: Request<EmbedRequest>) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        info!("Embed request for model: {}", req.model_id);

        // Load model
        let _ = self
            .model_manager
            .load_model(&req.model_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to load model: {}", e)))?;

        // Simulate embedding generation
        let embeddings = generate_embeddings(&req.input_data);
        let dimension = embeddings.len() as u32;

        // Generate proof
        let proof_of_service = self.proof_generator.generate_proof(&req.model_id, &req.query_id, start.elapsed().as_millis() as u64);

        let latency_ms = start.elapsed().as_millis() as u64;

        let response =
            EmbedResponse { embeddings, dimension, proof_of_service, model_version: "1".to_string(), latency_ms, signature: vec![] };

        info!("Embed completed in {}ms", latency_ms);
        Ok(Response::new(response))
    }

    #[instrument(skip(self, _request))]
    async fn get_model_info(&self, _request: Request<ModelInfoRequest>) -> Result<Response<ModelInfoResponse>, Status> {
        // Implementation would fetch actual model info
        let response = ModelInfoResponse {
            model_id: "default".to_string(),
            name: "Default Model".to_string(),
            description: "Default inference model".to_string(),
            version: "1.0".to_string(),
            category: "NLP".to_string(),
            model_hash: vec![0u8; 32],
            total_queries: 0,
            active: true,
            verified: true,
            last_updated: chrono::Utc::now().timestamp() as u64,
            metadata: Default::default(),
        };

        Ok(Response::new(response))
    }

    #[instrument(skip(self, _request))]
    async fn list_models(&self, _request: Request<ListModelsRequest>) -> Result<Response<ListModelsResponse>, Status> {
        let models = self.model_manager.list_models().await.map_err(|e| Status::internal(format!("Failed to list models: {}", e)))?;

        let model_infos: Vec<ModelInfo> = models
            .iter()
            .map(|m| ModelInfo {
                model_id: m.id.clone(),
                name: m.name.clone(),
                version: m.version.to_string(),
                category: m.category.clone(),
                active: m.loaded,
                verified: true,
            })
            .collect();

        let response = ListModelsResponse { models: model_infos, total_count: models.len() as u32 };

        Ok(Response::new(response))
    }

    #[instrument(skip(self, _request))]
    async fn health_check(&self, _request: Request<HealthCheckRequest>) -> Result<Response<HealthCheckResponse>, Status> {
        let response = HealthCheckResponse {
            healthy: true,
            version: "0.1.0".to_string(),
            uptime_seconds: 0,
            metrics: Default::default(),
            services: vec!["inference".to_string(), "storage".to_string()],
        };

        Ok(Response::new(response))
    }
}

impl InferenceService {
    pub fn into_server(self) -> InferenceServer<Self> {
        InferenceServer::new(self)
    }
}

fn simulate_inference(input_data: &[u8], _model_id: &str) -> Vec<u8> {
    // Simulate inference by processing input
    let mut output = Vec::with_capacity(input_data.len());
    for byte in input_data {
        output.push(byte.wrapping_add(1));
    }
    output
}

fn generate_embeddings(input_data: &[u8]) -> Vec<f32> {
    // Generate mock embeddings (384-dimensional)
    let dimension = 384;
    let mut embeddings = Vec::with_capacity(dimension);
    for i in 0..dimension {
        let value = if i < input_data.len() { input_data[i] as f32 / 255.0 } else { rand::random::<f32>() };
        embeddings.push(value);
    }
    embeddings
}
