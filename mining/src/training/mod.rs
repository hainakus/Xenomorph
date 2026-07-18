//! Training-based mining module for UsefulPoW

pub mod miner;
pub mod model_trainer;

pub use miner::{TrainingMiner, TrainingMinerConfig, TrainingJobManager};
pub use model_trainer::{ModelTrainer, TrainingBatch, TrainingResult, TrainingJobManager as AITrainingJobManager};
