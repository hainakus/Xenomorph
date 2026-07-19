//! Proof of Work module
//!
//! This module contains the training proof system for UsefulPoW.

pub mod training_proof;

pub use training_proof::{DifficultyTarget, ModelId, PublicInputs, TrainingProof, ZKProof};
