#![allow(dead_code)]

use anyhow::{Context, Result};
use axum::{
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::EnvFilter;

mod governance;
mod handlers;
mod payments;
mod proto;
mod seed_client;
mod state;

use governance::GovernanceClient;
use payments::verifier::PaymentVerifier;
use state::AppState;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into())).init();

    info!("Starting Xenomorph API Gateway");

    // Initialize payment verifier
    let payment_verifier = Arc::new(
        PaymentVerifier::new(
            std::env::var("USDT_CONTRACT_ADDRESS").unwrap_or_else(|_| "0x0000000000000000000000000000000000000000".to_string()),
            std::env::var("RPC_URL").unwrap_or_else(|_| "https://polygon-mumbai.infura.io/v3/YOUR_KEY".to_string()),
        )
        .await?,
    );

    // Initialize on-chain governance client
    let governance_contract = std::env::var("GOVERNANCE_CONTRACT_ADDRESS")
        .unwrap_or_else(|_| "0x0000000000000000000000000000000000000000".to_string())
        .parse()
        .context("Invalid GOVERNANCE_CONTRACT_ADDRESS")?;
    let governance_rpc = std::env::var("GOVERNANCE_RPC_URL")
        .unwrap_or_else(|_| "https://polygon-mumbai.infura.io/v3/YOUR_KEY".to_string());
    let governance_key = std::env::var("GOVERNANCE_OPERATOR_KEY").ok();
    let governance = Arc::new(GovernanceClient::new(
        &governance_rpc,
        governance_contract,
        governance_key.as_deref(),
    )?);

    let state = Arc::new(AppState::new(payment_verifier, governance).await?);

    // Build router
    let app = Router::new()
        .route("/models", get(handlers::models::list_models))
        .route("/models/{id}", get(handlers::models::get_model))
        .route("/predict/{model_id}", post(handlers::predict::predict))
        .route("/queries/{id}", get(handlers::predict::get_query_status))
        .route("/webhook/payment", post(handlers::predict::payment_webhook))
        // Governance endpoints
        .route("/governance/proposals", get(governance::proposals::list_proposals).post(governance::proposals::create_proposal))
        .route("/governance/proposals/:id", get(governance::proposals::get_proposal))
        .route("/governance/proposals/:id/vote", post(governance::voting::cast_vote))
        .route("/governance/proposals/:id/execute", post(governance::voting::execute_proposal))
        .route("/governance/models", get(governance::proposals::list_active_models))
        .route("/governance/models/:id", get(governance::proposals::get_active_model))
        .route("/health", get(health_check))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state);

    // Start server
    let addr: SocketAddr = "0.0.0.0:3000".parse()?;
    info!("API Gateway listening on {}", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> Result<Json<serde_json::Value>, StatusCode> {
    Ok(Json(serde_json::json!({
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}
