use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info, instrument};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: u32,
    pub prompt_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<OpenAiModel>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiModel {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[instrument(skip(state, request))]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionsRequest>,
) -> Result<Json<ChatCompletionsResponse>, StatusCode> {
    if request.stream == Some(true) {
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    // Use the last user message as the DNA query.
    let user_content =
        request.messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone()).ok_or(StatusCode::BAD_REQUEST)?;

    info!("OpenAI chat completion for model: {}", request.model);

    let model_id = request.model.clone();
    let response = if let Some(mut client) = state.seed_client.clone() {
        let query_id = Uuid::new_v4().to_string();
        match client.predict(&model_id, user_content.as_bytes(), &query_id).await {
            Ok(grpc_response) => String::from_utf8_lossy(&grpc_response.output_data).to_string(),
            Err(e) => {
                error!("Seed node predict failed for OpenAI chat: {}", e);
                return Err(StatusCode::BAD_GATEWAY);
            }
        }
    } else {
        error!("No seed node configured for OpenAI chat completions");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    let prompt_tokens = user_content.split_whitespace().count() as u32;
    let completion_tokens = response.split_whitespace().count() as u32;

    Ok(Json(ChatCompletionsResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: request.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage { role: "assistant".to_string(), content: response },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
    }))
}

#[instrument(skip(state, request))]
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, StatusCode> {
    let inputs = match request.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Multiple(v) => v,
    };

    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0u32;

    if let Some(mut client) = state.seed_client.clone() {
        for (index, input) in inputs.into_iter().enumerate() {
            let query_id = Uuid::new_v4().to_string();
            let prompt_tokens = input.split_whitespace().count() as u32;
            total_tokens += prompt_tokens;

            match client.embed(&request.model, input.as_bytes(), &query_id).await {
                Ok(grpc_response) => {
                    data.push(EmbeddingData {
                        object: "embedding".to_string(),
                        embedding: grpc_response.embeddings,
                        index: index as u32,
                        prompt_tokens,
                    });
                }
                Err(e) => {
                    error!("Seed node embed failed for OpenAI embeddings: {}", e);
                    return Err(StatusCode::BAD_GATEWAY);
                }
            }
        }
    } else {
        error!("No seed node configured for OpenAI embeddings");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    Ok(Json(EmbeddingsResponse {
        object: "list".to_string(),
        data,
        model: request.model,
        usage: Usage { prompt_tokens: total_tokens, completion_tokens: 0, total_tokens },
    }))
}

#[instrument(skip(state))]
pub async fn list_models(State(state): State<Arc<AppState>>) -> Result<Json<ModelsResponse>, StatusCode> {
    let models = if let Some(mut client) = state.seed_client.clone() {
        match client.list_models().await {
            Ok(grpc_response) => grpc_response
                .models
                .into_iter()
                .map(|m| OpenAiModel {
                    id: m.model_id,
                    object: "model".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    owned_by: "xenom".to_string(),
                })
                .collect(),
            Err(e) => {
                error!("Seed node list_models failed: {}", e);
                return Err(StatusCode::BAD_GATEWAY);
            }
        }
    } else {
        error!("No seed node configured for OpenAI models list");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    Ok(Json(ModelsResponse { object: "list".to_string(), data: models }))
}

pub async fn get_model(State(state): State<Arc<AppState>>, Path(model_id): Path<String>) -> Result<Json<OpenAiModel>, StatusCode> {
    let models = if let Some(mut client) = state.seed_client.clone() {
        match client.list_models().await {
            Ok(grpc_response) => grpc_response.models,
            Err(e) => {
                error!("Seed node list_models failed: {}", e);
                return Err(StatusCode::BAD_GATEWAY);
            }
        }
    } else {
        error!("No seed node configured for OpenAI model retrieval");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    let model = models.into_iter().find(|m| m.model_id == model_id).ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(OpenAiModel {
        id: model.model_id,
        object: "model".to_string(),
        created: chrono::Utc::now().timestamp(),
        owned_by: "xenom".to_string(),
    }))
}
