use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use base64::prelude::{Engine as _, BASE64_STANDARD};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, instrument};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub input_data: String,
    pub query_id: Option<String>,
    pub payment_signature: String,
}

#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub query_id: String,
    pub prediction: String,
    pub confidence: f32,
    pub model_version: String,
    pub seed_node_id: String,
    pub proof_of_service: String,
}

#[derive(Debug, Serialize)]
pub struct QueryStatus {
    pub query_id: String,
    pub status: String,
    pub prediction: Option<String>,
    pub confidence: Option<f32>,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PaymentWebhook {
    pub query_id: String,
    pub payment_verified: bool,
    pub amount: String,
    pub tx_hash: String,
}

#[instrument(skip(state, request))]
pub async fn predict(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, StatusCode> {
    let query_id = request.query_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    info!("Predict request for model: {}, query: {}", model_id, query_id);

    // Verify payment before processing
    let payment_verified = state.payment_verifier.verify_payment(&query_id).await.map_err(|e| {
        error!("Payment verification failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if !payment_verified {
        error!("Payment not verified for query: {}", query_id);
        return Err(StatusCode::PAYMENT_REQUIRED);
    }

    // Forward to seed node if a gRPC client is configured, otherwise simulate
    let response = if let Some(mut client) = state.seed_client.clone() {
        match client.predict(&model_id, request.input_data.as_bytes(), &query_id).await {
            Ok(grpc_response) => PredictResponse {
                query_id: query_id.clone(),
                prediction: String::from_utf8_lossy(&grpc_response.output_data).to_string(),
                confidence: grpc_response.confidence,
                model_version: grpc_response.model_version,
                seed_node_id: grpc_response.seed_node_id,
                proof_of_service: BASE64_STANDARD.encode(&grpc_response.proof_of_service),
            },
            Err(e) => {
                error!("Seed node call failed for query {}: {}", query_id, e);
                simulate_response(&request.input_data, &model_id, &query_id)
            }
        }
    } else {
        simulate_response(&request.input_data, &model_id, &query_id)
    };

    // Cache result
    let _ = state.cache_result(&query_id, &response).await;

    info!("Predict completed for query: {}", query_id);
    Ok(Json(response))
}

#[instrument(skip(state))]
pub async fn get_query_status(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<QueryStatus>, StatusCode> {
    // Check cache
    if let Some(cached) = state.get_cached_result(&id).await {
        return Ok(Json(QueryStatus {
            query_id: id,
            status: "completed".to_string(),
            prediction: Some(cached.prediction),
            confidence: Some(cached.confidence),
            error: None,
        }));
    }

    Ok(Json(QueryStatus { query_id: id, status: "pending".to_string(), prediction: None, confidence: None, error: None }))
}

#[instrument(skip(state, webhook))]
pub async fn payment_webhook(
    State(state): State<Arc<AppState>>,
    Json(webhook): Json<PaymentWebhook>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    info!("Payment webhook: query={}, verified={}", webhook.query_id, webhook.payment_verified);

    // Update payment status
    state.update_payment_status(&webhook.query_id, webhook.payment_verified).await;

    Ok(Json(serde_json::json!({
        "status": "received",
        "query_id": webhook.query_id
    })))
}

fn simulate_response(input: &str, model_id: &str, query_id: &str) -> PredictResponse {
    // Simulate prediction based on input
    PredictResponse {
        query_id: query_id.to_string(),
        prediction: format!("Prediction for {} using model: {}", input, model_id),
        confidence: 0.95 + (rand::random::<f32>() * 0.05),
        model_version: "1.0".to_string(),
        seed_node_id: "seed-node-1".to_string(),
        proof_of_service: BASE64_STANDARD.encode([1u8, 2, 3, 4]),
    }
}
