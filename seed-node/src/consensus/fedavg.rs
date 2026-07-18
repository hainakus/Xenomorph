use anyhow::Result;
use borsh::{BorshDeserialize, BorshSerialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FedAvgError {
    #[error("No gradients to aggregate")]
    NoGradients,

    #[error("Gradient shape mismatch")]
    ShapeMismatch,

    #[error("Insufficient participants")]
    InsufficientParticipants,

    #[error("Weight calculation error")]
    WeightError,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct GradientAccumulator {
    pub layer_name: String,
    pub accumulated_gradients: Vec<f32>,
    pub shape: Vec<usize>,
    pub participant_count: u32,
    pub total_weight: f32,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct FedAvgConfig {
    pub min_participants: u32,
    pub max_participants: u32,
    pub weighting_strategy: WeightingStrategy,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub enum WeightingStrategy {
    Uniform,
    ByDatasetSize,
    ByLossImprovement,
}

impl Default for FedAvgConfig {
    fn default() -> Self {
        Self { min_participants: 2, max_participants: 10, weighting_strategy: WeightingStrategy::Uniform }
    }
}

pub struct FedAvgAggregator {
    config: FedAvgConfig,
    accumulators: HashMap<String, GradientAccumulator>,
}

impl FedAvgAggregator {
    pub fn new(config: FedAvgConfig) -> Self {
        Self { config, accumulators: HashMap::new() }
    }

    pub fn with_default_config() -> Self {
        Self::new(FedAvgConfig::default())
    }

    pub fn add_gradient(&mut self, layer_name: &str, gradient: Vec<f32>, shape: Vec<usize>, weight: f32) -> Result<(), FedAvgError> {
        if gradient.is_empty() {
            return Err(FedAvgError::NoGradients);
        }

        if shape.is_empty() {
            return Err(FedAvgError::ShapeMismatch);
        }

        let accumulator = self.accumulators.entry(layer_name.to_string()).or_insert_with(|| GradientAccumulator {
            layer_name: layer_name.to_string(),
            accumulated_gradients: vec![0.0; gradient.len()],
            shape: shape.clone(),
            participant_count: 0,
            total_weight: 0.0,
        });

        // Verify shape matches
        if accumulator.shape != shape {
            return Err(FedAvgError::ShapeMismatch);
        }

        if accumulator.accumulated_gradients.len() != gradient.len() {
            return Err(FedAvgError::ShapeMismatch);
        }

        // Add weighted gradient
        for (i, grad) in gradient.iter().enumerate() {
            accumulator.accumulated_gradients[i] += grad * weight;
        }

        accumulator.participant_count += 1;
        accumulator.total_weight += weight;

        Ok(())
    }

    pub fn compute_average(&mut self, layer_name: &str) -> Result<Vec<f32>, FedAvgError> {
        let accumulator = self.accumulators.get(layer_name).ok_or(FedAvgError::NoGradients)?;

        if accumulator.participant_count < self.config.min_participants {
            return Err(FedAvgError::InsufficientParticipants);
        }

        if accumulator.total_weight <= 0.0 {
            return Err(FedAvgError::WeightError);
        }

        let averaged: Vec<f32> = accumulator.accumulated_gradients.iter().map(|g| g / accumulator.total_weight).collect();

        Ok(averaged)
    }

    pub fn compute_all_averages(&mut self) -> Result<HashMap<String, Vec<f32>>, FedAvgError> {
        let mut result = HashMap::new();

        for layer_name in self.accumulators.keys().cloned().collect::<Vec<_>>() {
            let average = self.compute_average(&layer_name)?;
            result.insert(layer_name, average);
        }

        Ok(result)
    }

    pub fn reset(&mut self) {
        self.accumulators.clear();
    }

    pub fn participant_count(&self, layer_name: &str) -> u32 {
        self.accumulators.get(layer_name).map(|a| a.participant_count).unwrap_or(0)
    }

    pub fn is_ready(&self, layer_name: &str) -> bool {
        self.participant_count(layer_name) >= self.config.min_participants
    }

    pub fn all_ready(&self) -> bool {
        self.accumulators.values().all(|a| a.participant_count >= self.config.min_participants)
    }

    pub fn get_layer_info(&self, layer_name: &str) -> Option<&GradientAccumulator> {
        self.accumulators.get(layer_name)
    }

    pub fn serialize_state(&self) -> Result<Vec<u8>, FedAvgError> {
        let state: Vec<&GradientAccumulator> = self.accumulators.values().collect();
        state.try_to_vec().map_err(|_| FedAvgError::NoGradients)
    }

    pub fn deserialize_state(&mut self, data: &[u8]) -> Result<(), FedAvgError> {
        let accumulators: Vec<GradientAccumulator> = Vec::try_from_slice(data).map_err(|_| FedAvgError::NoGradients)?;

        for accumulator in accumulators {
            self.accumulators.insert(accumulator.layer_name.clone(), accumulator);
        }

        Ok(())
    }
}

impl Default for FedAvgAggregator {
    fn default() -> Self {
        Self::with_default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_aggregation() {
        let mut aggregator = FedAvgAggregator::with_default_config();

        let gradient1 = vec![1.0, 2.0, 3.0];
        let gradient2 = vec![2.0, 4.0, 6.0];
        let shape = vec![3];

        aggregator.add_gradient("layer1", gradient1, shape.clone(), 1.0).unwrap();
        aggregator.add_gradient("layer1", gradient2, shape, 1.0).unwrap();

        let average = aggregator.compute_average("layer1").unwrap();
        assert_eq!(average, vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_insufficient_participants() {
        let aggregator = FedAvgAggregator::with_default_config();
        let gradient = vec![1.0, 2.0, 3.0];
        let shape = vec![3];

        let mut agg = aggregator;
        agg.add_gradient("layer1", gradient, shape, 1.0).unwrap();

        let result = agg.compute_average("layer1");
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let mut aggregator = FedAvgAggregator::with_default_config();

        let gradient1 = vec![1.0, 2.0, 3.0];
        let gradient2 = vec![4.0, 5.0];
        let shape1 = vec![3];
        let shape2 = vec![2];

        aggregator.add_gradient("layer1", gradient1, shape1, 1.0).unwrap();
        let result = aggregator.add_gradient("layer1", gradient2, shape2, 1.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_aggregation() {
        let mut aggregator = FedAvgAggregator::with_default_config();

        let gradient1 = vec![1.0, 2.0, 3.0];
        let gradient2 = vec![2.0, 4.0, 6.0];
        let shape = vec![3];

        aggregator.add_gradient("layer1", gradient1, shape.clone(), 2.0).unwrap();
        aggregator.add_gradient("layer1", gradient2, shape, 1.0).unwrap();

        let average = aggregator.compute_average("layer1").unwrap();
        // (1*2 + 2*1) / 3 = 1.33, (2*2 + 4*1) / 3 = 2.67, (3*2 + 6*1) / 3 = 4.0
        assert!((average[0] - 1.33).abs() < 0.01);
        assert!((average[1] - 2.67).abs() < 0.01);
        assert!((average[2] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut aggregator = FedAvgAggregator::with_default_config();

        let gradient = vec![1.0, 2.0, 3.0];
        let shape = vec![3];

        aggregator.add_gradient("layer1", gradient, shape, 1.0).unwrap();
        aggregator.reset();

        assert_eq!(aggregator.participant_count("layer1"), 0);
    }
}
