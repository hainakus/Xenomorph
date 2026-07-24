use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};
use tracing::{info, instrument};

use crate::model::manager::ModelManager;
use crate::rpc::client::XenomorphRpcClient;
use crate::serving::inference_engine::InferenceEngine;
use crate::serving::proof::ProofGenerator;

pub mod xenom {
    pub mod inference {
        tonic::include_proto!("xenom.inference");
    }
}

use xenom::inference::inference_server::{Inference, InferenceServer};
use xenom::inference::*;

pub struct InferenceService {
    engine: Arc<InferenceEngine>,
    xenomorph_client: Arc<XenomorphRpcClient>,
    proof_generator: ProofGenerator,
}

impl InferenceService {
    pub fn new(model_manager: Arc<ModelManager>, xenomorph_client: Arc<XenomorphRpcClient>) -> Self {
        Self { engine: Arc::new(InferenceEngine::new(model_manager)), xenomorph_client, proof_generator: ProofGenerator::new() }
    }
}

#[tonic::async_trait]
impl Inference for InferenceService {
    #[instrument(skip(self, request))]
    async fn predict(&self, request: Request<PredictRequest>) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();
        let model_id = req.model_id.clone();
        let query_id = req.query_id.clone();

        info!("Predict request for model: {}", model_id);

        let input = String::from_utf8_lossy(&req.input_data).to_string();
        let engine = self.engine.clone();
        let (output, confidence) = tokio::task::spawn_blocking(move || engine.predict(&model_id, &input))
            .await
            .map_err(|e| Status::internal(format!("Inference task panicked: {}", e)))?
            .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

        let latency_ms = start.elapsed().as_millis() as u64;
        let proof_of_service = self.proof_generator.generate_proof(&req.model_id, &query_id, latency_ms);

        let response = PredictResponse {
            output_data: output.into_bytes(),
            confidence,
            proof_of_service,
            model_version: "1".to_string(),
            latency_ms,
            seed_node_id: self.engine.node_id().to_string(),
            signature: vec![],
        };

        info!("Predict completed in {}ms", latency_ms);
        Ok(Response::new(response))
    }

    #[instrument(skip(self, request))]
    async fn embed(&self, request: Request<EmbedRequest>) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();
        let model_id = req.model_id.clone();

        info!("Embed request for model: {}", model_id);

        let input = String::from_utf8_lossy(&req.input_data).to_string();
        let engine = self.engine.clone();
        let embeddings = tokio::task::spawn_blocking(move || engine.embed(&model_id, &input))
            .await
            .map_err(|e| Status::internal(format!("Inference task panicked: {}", e)))?
            .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

        let dimension = embeddings.len() as u32;
        let latency_ms = start.elapsed().as_millis() as u64;
        let proof_of_service = self.proof_generator.generate_proof(&req.model_id, &req.query_id, latency_ms);

        let response =
            EmbedResponse { embeddings, dimension, proof_of_service, model_version: "1".to_string(), latency_ms, signature: vec![] };

        info!("Embed completed in {}ms", latency_ms);
        Ok(Response::new(response))
    }

    #[instrument(skip(self, request))]
    async fn get_model_info(&self, request: Request<ModelInfoRequest>) -> Result<Response<ModelInfoResponse>, Status> {
        let req = request.into_inner();
        let model_id = req.model_id;

        let model_info = self
            .engine
            .model_manager()
            .get_model(&model_id)
            .await
            .ok_or_else(|| Status::not_found(format!("Model {} not found", model_id)))?;

        let response = ModelInfoResponse {
            model_id: model_info.id,
            name: model_info.name,
            description: format!("{} sequence analysis model", model_info.category),
            version: model_info.version.to_string(),
            category: model_info.category,
            model_hash: model_info.checkpoint.weights_hash.to_vec(),
            total_queries: model_info.last_used,
            active: model_info.loaded,
            verified: true,
            last_updated: model_info.last_used,
            metadata: Default::default(),
        };

        Ok(Response::new(response))
    }

    #[instrument(skip(self, _request))]
    async fn list_models(&self, _request: Request<ListModelsRequest>) -> Result<Response<ListModelsResponse>, Status> {
        let models =
            self.engine.model_manager().list_models().await.map_err(|e| Status::internal(format!("Failed to list models: {}", e)))?;

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
