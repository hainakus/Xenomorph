use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};
use tracing::{info, instrument};

use crate::model::manager::ModelManager;
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
    proof_generator: ProofGenerator,
}

impl InferenceService {
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self { engine: Arc::new(InferenceEngine::new(model_manager)), proof_generator: ProofGenerator::new() }
    }

    /// If `block_height` is non-zero, ensure the requested historical checkpoint
    /// is loaded before inference.  Returns an error if the checkpoint is unknown.
    async fn ensure_historical_checkpoint(&self, model_id: &str, block_height: u64) -> Result<(), Status> {
        if block_height == 0 {
            return Ok(());
        }

        let manager = self.engine.model_manager();
        let current = manager.active_hash(model_id).await.ok_or_else(|| Status::not_found(format!("Model {} not found", model_id)))?;

        // Look up the checkpoint that was active at the requested block.
        let historical_hash = manager
            .checkpoint_at(block_height)
            .await
            .ok_or_else(|| Status::not_found(format!("No checkpoint recorded for model {} at block {}", model_id, block_height)))?;

        // Already on the right checkpoint (active or previously loaded).
        if historical_hash == current {
            return Ok(());
        }

        // Verify the historical checkpoint belongs to the active model lineage.
        if !manager.is_ancestor_of_active(model_id, historical_hash).await {
            return Err(Status::not_found(format!(
                "Checkpoint at block {} is not in the active lineage for {}",
                block_height, model_id
            )));
        }

        manager
            .load_historical_checkpoint(model_id, historical_hash)
            .await
            .map_err(|e| Status::internal(format!("Failed to load historical checkpoint: {}", e)))?;

        Ok(())
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

        self.ensure_historical_checkpoint(&model_id, req.block_height).await?;

        let input = String::from_utf8_lossy(&req.input_data).to_string();
        let engine = self.engine.clone();
        let (output, confidence, prompt_tokens, completion_tokens) =
            tokio::task::spawn_blocking(move || engine.predict(&model_id, &input))
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
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: completion_tokens as u32,
        };

        info!("Predict completed in {}ms", latency_ms);
        Ok(Response::new(response))
    }

    #[instrument(skip(self, request))]
    async fn evaluate_masked_llm(
        &self,
        request: Request<EvaluateMaskedLlmRequest>,
    ) -> Result<Response<EvaluateMaskedLlmResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();
        let model_id = req.model_id.clone();

        info!("EvaluateMaskedLlm request for model: {}", model_id);

        self.ensure_historical_checkpoint(&model_id, req.block_height).await?;

        let input = String::from_utf8_lossy(&req.input_data).to_string();
        let engine = self.engine.clone();
        let result = tokio::task::spawn_blocking(move || engine.evaluate_masked_lm(&model_id, &input))
            .await
            .map_err(|e| Status::internal(format!("Inference task panicked: {}", e)))?
            .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;

        let latency_ms = start.elapsed().as_millis() as u64;

        let response = EvaluateMaskedLlmResponse {
            output_data: result.output.into_bytes(),
            confidence: result.confidence,
            prompt_tokens: result.prompt_tokens as u32,
            completion_tokens: result.completion_tokens as u32,
            masked_positions: result.masked_positions,
            masked_logits: result.masked_logits,
            logits_vocab_size: result.logits_vocab_size,
            model_version: "1".to_string(),
            latency_ms,
        };

        info!("EvaluateMaskedLlm completed in {}ms", latency_ms);
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
        let block_height = req.block_height;

        let manager = self.engine.model_manager();

        // If a block height is provided, look up the checkpoint that was active
        // at that point in the training chain and verify it is still loadable.
        let model_info = if block_height > 0 {
            let historical_hash = manager
                .checkpoint_at(block_height)
                .await
                .ok_or_else(|| Status::not_found(format!("No checkpoint recorded for {} at block {}", model_id, block_height)))?;

            // Ensure the model id is known and the historical hash is part of its lineage.
            let current =
                manager.get_model(&model_id).await.ok_or_else(|| Status::not_found(format!("Model {} not found", model_id)))?;

            if historical_hash != current.checkpoint.weights_hash && !manager.is_ancestor_of_active(&model_id, historical_hash).await {
                return Err(Status::not_found(format!(
                    "Checkpoint {} at block {} is not part of the active lineage for {}",
                    hex::encode(historical_hash),
                    block_height,
                    model_id
                )));
            }

            // Load the requested historical checkpoint if it is not the active one.
            if historical_hash != current.checkpoint.weights_hash {
                manager
                    .load_historical_checkpoint(&model_id, historical_hash)
                    .await
                    .map_err(|e| Status::internal(format!("Failed to load historical checkpoint: {}", e)))?;
            }

            manager.get_model(&model_id).await.ok_or_else(|| Status::not_found(format!("Model {} not found", model_id)))?
        } else {
            manager.get_model(&model_id).await.ok_or_else(|| Status::not_found(format!("Model {} not found", model_id)))?
        };

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
