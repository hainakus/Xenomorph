use crate::proto::xenomorph::inference::{
    inference_client::InferenceClient, EmbedRequest as GrpcEmbedRequest, EmbedResponse as GrpcEmbedResponse,
    ListModelsRequest as GrpcListModelsRequest, ListModelsResponse as GrpcListModelsResponse,
    ModelInfoRequest as GrpcModelInfoRequest, ModelInfoResponse as GrpcModelInfoResponse, PredictRequest as GrpcPredictRequest,
    PredictResponse as GrpcPredictResponse,
};
use anyhow::{anyhow, Result};
use tonic::transport::Channel;

#[derive(Clone, Debug)]
pub struct SeedNodeClient {
    client: InferenceClient<Channel>,
}

impl SeedNodeClient {
    pub async fn connect(addr: &str) -> Result<Self> {
        let client = InferenceClient::<Channel>::connect(addr.to_string())
            .await
            .map_err(|e| anyhow!("Failed to connect to seed node: {}", e))?;
        Ok(Self { client })
    }

    pub async fn predict(
        &mut self,
        model_id: &str,
        input_data: &[u8],
        query_id: &str,
        block_height: Option<u64>,
    ) -> Result<GrpcPredictResponse> {
        let request = tonic::Request::new(GrpcPredictRequest {
            model_id: model_id.to_string(),
            input_data: input_data.to_vec(),
            query_id: query_id.to_string(),
            encryption_key: Vec::new(),
            metadata: Default::default(),
            block_height: block_height.unwrap_or(0),
        });

        let response = self.client.predict(request).await.map_err(|e| anyhow!("Seed node prediction failed: {}", e))?;

        Ok(response.into_inner())
    }

    pub async fn embed(&mut self, model_id: &str, input_data: &[u8], query_id: &str) -> Result<GrpcEmbedResponse> {
        let request = tonic::Request::new(GrpcEmbedRequest {
            model_id: model_id.to_string(),
            input_data: input_data.to_vec(),
            query_id: query_id.to_string(),
            encryption_key: Vec::new(),
        });

        let response = self.client.embed(request).await.map_err(|e| anyhow!("Seed node embed failed: {}", e))?;

        Ok(response.into_inner())
    }

    pub async fn list_models(&mut self) -> Result<GrpcListModelsResponse> {
        let request =
            tonic::Request::new(GrpcListModelsRequest { category: String::new(), active_only: false, limit: 100, offset: 0 });

        let response = self.client.list_models(request).await.map_err(|e| anyhow!("Seed node list_models failed: {}", e))?;

        Ok(response.into_inner())
    }

    pub async fn get_model_info(&mut self, model_id: &str, block_height: Option<u64>) -> Result<GrpcModelInfoResponse> {
        let request =
            tonic::Request::new(GrpcModelInfoRequest { model_id: model_id.to_string(), block_height: block_height.unwrap_or(0) });

        let response = self.client.get_model_info(request).await.map_err(|e| anyhow!("Seed node get_model_info failed: {}", e))?;

        Ok(response.into_inner())
    }
}
