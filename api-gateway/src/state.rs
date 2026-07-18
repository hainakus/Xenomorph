use crate::governance::GovernanceClient;
use crate::handlers::predict::PredictResponse;
use crate::payments::verifier::PaymentVerifier;
use crate::seed_client::SeedNodeClient;
use anyhow::{anyhow, Result};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    pub query_id: String,
    pub prediction: String,
    pub confidence: f32,
    pub model_version: String,
    pub seed_node_id: String,
    pub proof_of_service: String,
    pub cached_at: u64,
}

pub struct AppState {
    pub payment_verifier: Arc<PaymentVerifier>,
    pub governance: Arc<GovernanceClient>,
    pub redis_client: Arc<redis::Client>,
    pub cache: Arc<RwLock<HashMap<String, CachedResult>>>,
    pub payment_status: Arc<RwLock<HashMap<String, bool>>>,
    pub seed_client: Option<SeedNodeClient>,
}

impl AppState {
    pub async fn new(
        payment_verifier: Arc<PaymentVerifier>,
        governance: Arc<GovernanceClient>,
    ) -> Result<Self> {
        // Initialize Redis client
        let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

        let redis_client = redis::Client::open(redis_url).map_err(|e| anyhow!("Failed to create Redis client: {}", e))?;

        // Test Redis connection
        let mut conn: redis::aio::Connection =
            redis_client.get_async_connection().await.map_err(|e| anyhow!("Failed to connect to Redis: {}", e))?;

        // Ping Redis
        let _: () = redis::cmd("PING").query_async::<_, ()>(&mut conn).await.map_err(|e| anyhow!("Redis ping failed: {}", e))?;

        info!("Connected to Redis");

        // Initialize optional seed-node gRPC client
        let seed_client = match std::env::var("SEED_NODE_ADDR") {
            Ok(addr) => match SeedNodeClient::connect(&addr).await {
                Ok(client) => {
                    info!("Connected to seed node at {}", addr);
                    Some(client)
                }
                Err(e) => {
                    info!("Could not connect to seed node at {}: {}", addr, e);
                    None
                }
            },
            Err(_) => None,
        };

        Ok(Self {
            payment_verifier,
            governance,
            redis_client: Arc::new(redis_client),
            cache: Arc::new(RwLock::new(HashMap::new())),
            payment_status: Arc::new(RwLock::new(HashMap::new())),
            seed_client,
        })
    }

    pub async fn cache_result(&self, query_id: &str, response: &PredictResponse) -> Result<()> {
        let cached = CachedResult {
            query_id: query_id.to_string(),
            prediction: response.prediction.clone(),
            confidence: response.confidence,
            model_version: response.model_version.clone(),
            seed_node_id: response.seed_node_id.clone(),
            proof_of_service: response.proof_of_service.clone(),
            cached_at: chrono::Utc::now().timestamp() as u64,
        };

        // Store in memory cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(query_id.to_string(), cached.clone());
        }

        // Store in Redis
        let mut conn: redis::aio::Connection =
            self.redis_client.get_async_connection().await.map_err(|e| anyhow!("Failed to get Redis connection: {}", e))?;

        let key = format!("result:{}", query_id);
        let value = serde_json::to_string(&cached).map_err(|e| anyhow!("Failed to serialize result: {}", e))?;

        let _: () = conn.set_ex(&key, &value, 3600).await.map_err(|e| anyhow!("Failed to cache result in Redis: {}", e))?;

        info!("Cached result for query: {}", query_id);
        Ok(())
    }

    pub async fn get_cached_result(&self, query_id: &str) -> Option<CachedResult> {
        // Check memory cache first
        {
            let cache = self.cache.read().await;
            if let Some(result) = cache.get(query_id) {
                return Some(result.clone());
            }
        }

        // Check Redis
        if let Ok(mut conn) = self.redis_client.get_async_connection().await {
            let key = format!("result:{}", query_id);
            if let Ok(value) = conn.get::<_, String>(&key).await {
                if let Ok(cached) = serde_json::from_str::<CachedResult>(&value) {
                    // Update memory cache
                    let mut cache = self.cache.write().await;
                    cache.insert(query_id.to_string(), cached.clone());
                    return Some(cached);
                }
            }
        }

        None
    }

    pub async fn update_payment_status(&self, query_id: &str, verified: bool) {
        let mut status = self.payment_status.write().await;
        status.insert(query_id.to_string(), verified);

        // Also update Redis
        if let Ok(mut conn) = self.redis_client.get_async_connection().await {
            let key = format!("payment:{}", query_id);
            let _: Result<(), redis::RedisError> = conn.set_ex(&key, verified, 3600).await;
        }

        info!("Updated payment status for query {}: {}", query_id, verified);
    }

    pub async fn get_payment_status(&self, query_id: &str) -> Option<bool> {
        let status = self.payment_status.read().await;
        status.get(query_id).copied()
    }

    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        info!("Cache cleared");
        Ok(())
    }

    pub async fn cache_size(&self) -> usize {
        let cache = self.cache.read().await;
        cache.len()
    }
}
