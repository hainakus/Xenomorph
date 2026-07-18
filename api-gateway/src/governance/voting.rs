//! REST handlers for casting votes and executing proposals.

use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

#[derive(Debug, Deserialize)]
pub struct VoteRequest {
    pub support: bool,
}

#[derive(Debug, Serialize)]
pub struct VoteResponse {
    pub tx_hash: String,
}

pub async fn cast_vote(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
    Json(req): Json<VoteRequest>,
) -> Result<Json<VoteResponse>, StatusCode> {
    match state.governance.vote(id, req.support).await {
        Ok(tx_hash) => {
            info!("Cast vote on proposal {} via gateway", id);
            Ok(Json(VoteResponse {
                tx_hash: format!("{:#x}", tx_hash),
            }))
        }
        Err(e) => {
            error!("Failed to vote on proposal {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ExecuteResponse {
    pub tx_hash: String,
    pub activated: bool,
}

pub async fn execute_proposal(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<ExecuteResponse>, StatusCode> {
    match state.governance.execute(id).await {
        Ok(tx_hash) => {
            info!("Executed proposal {} via gateway", id);
            Ok(Json(ExecuteResponse {
                tx_hash: format!("{:#x}", tx_hash),
                activated: true,
            }))
        }
        Err(e) => {
            error!("Failed to execute proposal {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
