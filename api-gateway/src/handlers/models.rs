use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::Serialize;
use std::sync::Arc;

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

pub async fn list_models(State(_state): State<Arc<AppState>>) -> Result<Json<ModelsResponse>, StatusCode> {
    // In production, this would query the seed nodes or model registry
    let models = vec![
        ModelInfo {
            id: "dnabert2".to_string(),
            name: "DNABERT2".to_string(),
            description: "DNA sequence analysis model".to_string(),
            version: "1.0".to_string(),
            category: "NLP".to_string(),
            active: true,
            verified: true,
            total_queries: 1000,
        },
        ModelInfo {
            id: "protbert".to_string(),
            name: "ProtBERT".to_string(),
            description: "Protein sequence analysis model".to_string(),
            version: "1.0".to_string(),
            category: "NLP".to_string(),
            active: true,
            verified: true,
            total_queries: 500,
        },
    ];

    let total_count = models.len();
    Ok(Json(ModelsResponse { models, total_count }))
}

pub async fn get_model(State(_state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<ModelInfo>, StatusCode> {
    // In production, this would query the seed nodes or model registry
    if id == "dnabert2" {
        Ok(Json(ModelInfo {
            id: "dnabert2".to_string(),
            name: "DNABERT2".to_string(),
            description: "DNA sequence analysis model".to_string(),
            version: "1.0".to_string(),
            category: "NLP".to_string(),
            active: true,
            verified: true,
            total_queries: 1000,
        }))
    } else if id == "protbert" {
        Ok(Json(ModelInfo {
            id: "protbert".to_string(),
            name: "ProtBERT".to_string(),
            description: "Protein sequence analysis model".to_string(),
            version: "1.0".to_string(),
            category: "NLP".to_string(),
            active: true,
            verified: true,
            total_queries: 500,
        }))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}
