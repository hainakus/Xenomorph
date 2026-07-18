//! Zero-knowledge proof verification for UsefulPoW
//! 
//! This module handles ZK proof verification for training computations.

pub mod training_verifier;

pub use training_verifier::{ZKTrainingProof, verify_proof};
