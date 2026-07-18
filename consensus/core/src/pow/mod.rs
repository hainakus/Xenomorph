//! Proof of Work module
//! 
//! This module contains the training proof system for UsefulPoW.

pub mod training_proof;

pub use training_proof::{ModelId, DifficultyTarget, ZKProof, PublicInputs, TrainingProof};
