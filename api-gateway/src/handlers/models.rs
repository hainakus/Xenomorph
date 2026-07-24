use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::Serialize;
use std::sync::Arc;
use tracing::error;

#[derive(Debug, Serialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub category: String,
    pub active: bool,
    pub verified: bool,
    pub total_queries: u64,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub models: Vec<ModelInfo>,
    pub total_count: usize,
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Result<Json<ModelsResponse>, StatusCode> {
    let mut client = state.seed_client.clone().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let list_response = client.list_models().await.map_err(|e| {
        error!("Seed node list_models failed: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    let mut models = Vec::with_capacity(list_response.models.len());
    for grpc_model in list_response.models {
        // Augment the summary with full model info for description and query count.
        let details = client.get_model_info(&grpc_model.model_id).await.unwrap_or_default();

        let name = if details.name.is_empty() { grpc_model.name } else { details.name };
        let description = if details.description.is_empty() { "Xenomorph scientific model".to_string() } else { details.description };
        let version = if details.version.is_empty() { grpc_model.version } else { details.version };
        let category = if details.category.is_empty() { grpc_model.category } else { details.category };

        models.push(ModelInfo {
            id: grpc_model.model_id,
            name,
            description,
            version,
            category,
            active: grpc_model.active || details.active,
            verified: grpc_model.verified || details.verified,
            total_queries: details.total_queries,
        });
    }

    let total_count = models.len();
    Ok(Json(ModelsResponse { models, total_count }))
}

pub async fn get_model(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<ModelInfo>, StatusCode> {
    let mut client = state.seed_client.clone().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    let details = client.get_model_info(&id).await.map_err(|e| {
        error!("Seed node get_model_info failed for {}: {}", id, e);
        StatusCode::NOT_FOUND
    })?;

    Ok(Json(ModelInfo {
        id: details.model_id.clone(),
        name: if details.name.is_empty() { details.model_id } else { details.name },
        description: if details.description.is_empty() { "Xenomorph scientific model".to_string() } else { details.description },
        version: details.version,
        category: details.category,
        active: details.active,
        verified: details.verified,
        total_queries: details.total_queries,
    }))
}
