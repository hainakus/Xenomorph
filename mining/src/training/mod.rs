//! Training-based mining module for UsefulPoW

pub mod miner;
pub mod model_trainer;

pub use miner::{TrainingJobManager, TrainingMiner, TrainingMinerConfig};
pub use model_trainer::{ModelTrainer, TrainingBatch, TrainingJobManager as AITrainingJobManager, TrainingResult};
