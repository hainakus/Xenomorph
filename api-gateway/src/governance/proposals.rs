//! REST handlers for model proposals and active model listings.

use crate::state::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

#[derive(Debug, Deserialize)]
pub struct ListProposalsQuery {
    #[serde(default)]
    active_only: bool,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_limit() -> usize {
    100
}

#[derive(Debug, Serialize)]
pub struct ProposalsResponse {
    pub proposals: Vec<crate::governance::ProposalSummary>,
    pub total_count: usize,
}

#[derive(Debug, Serialize)]
pub struct ProposalResponse {
    pub proposal: crate::governance::ProposalSummary,
}

#[derive(Debug, Serialize)]
pub struct CreateProposalResponse {
    pub proposal_id: u64,
    pub tx_hash: String,
}

pub async fn list_proposals(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListProposalsQuery>,
) -> Result<Json<ProposalsResponse>, StatusCode> {
    match state.governance.list_proposals(query.active_only).await {
        Ok(mut proposals) => {
            let total_count = proposals.len();
            proposals = proposals
                .into_iter()
                .skip(query.offset)
                .take(query.limit)
                .collect();
            Ok(Json(ProposalsResponse {
                proposals,
                total_count,
            }))
        }
        Err(e) => {
            error!("Failed to list proposals: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn get_proposal(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<ProposalResponse>, StatusCode> {
    match state.governance.get_proposal(id).await {
        Ok(proposal) => Ok(Json(ProposalResponse { proposal })),
        Err(e) => {
            error!("Failed to get proposal {}: {}", id, e);
            Err(StatusCode::NOT_FOUND)
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateProposalRequest {
    pub model_id: String,
    pub hf_repo: String,
    pub hf_revision: String,
    pub genesis_checkpoint: String, // hex, 32 bytes
    pub vram_required: u64,
    pub reward_per_block: u64,
    pub min_stake_to_train: u64,
}

pub async fn create_proposal(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateProposalRequest>,
) -> Result<Json<CreateProposalResponse>, StatusCode> {
    let checkpoint = hex::decode(req.genesis_checkpoint.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    if checkpoint.len() != 32 {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&checkpoint);

    let governance_req = crate::governance::ProposeModelRequest {
        model_id: req.model_id,
        hf_repo: req.hf_repo,
        hf_revision: req.hf_revision,
        genesis_checkpoint: bytes,
        vram_required: req.vram_required,
        reward_per_block: req.reward_per_block,
        min_stake_to_train: req.min_stake_to_train,
    };

    match state.governance.propose_model(governance_req).await {
        Ok((tx_hash, proposal_id)) => {
            info!("Created proposal {} via gateway", proposal_id);
            Ok(Json(CreateProposalResponse {
                proposal_id,
                tx_hash: format!("{:#x}", tx_hash),
            }))
        }
        Err(e) => {
            error!("Failed to create proposal: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ActiveModelsResponse {
    pub models: Vec<crate::governance::ActiveModel>,
    pub total_count: usize,
}

pub async fn list_active_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ActiveModelsResponse>, StatusCode> {
    match state.governance.list_active_models().await {
        Ok(models) => {
            let total_count = models.len();
            Ok(Json(ActiveModelsResponse { models, total_count }))
        }
        Err(e) => {
            error!("Failed to list active models: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn get_active_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<crate::governance::ActiveModel>, StatusCode> {
    match state.governance.get_model_data(&id).await {
        Ok(Some(model)) => Ok(Json(model)),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            error!("Failed to get model {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
