//! ZK proof validation system for UsefulPoW
//!
//! This module implements the validator selection, ZK proof verification,
//! and validation network protocol for the UsefulPoW system.

pub mod config;
pub mod network;
pub mod selector;
pub mod zk_verifier;

pub use config::{ConfigError as ValidationError, ValidationParams};
pub use network::{
    AggregatedSignature, NetworkError, NetworkEvent, SignatureAggregator, ValidationMessage, ValidationNetwork, ValidationSignature,
};
pub use selector::{SelectionError, ValidatorInfo, ValidatorSelection, ValidatorSelectionSimple, ValidatorSelector};
pub use zk_verifier::{MockVerifier, VerificationMetadata, VerificationResult, ZKTrainingProof, ZKVerifier};
