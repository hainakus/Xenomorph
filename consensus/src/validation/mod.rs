//! ZK proof validation system for UsefulPoW
//!
//! This module implements the validator selection, ZK proof verification,
//! and validation network protocol for the UsefulPoW system.

pub mod selector;
pub mod zk_verifier;
pub mod network;
pub mod config;

pub use selector::{ValidatorSelector, ValidatorInfo, ValidatorSelection, ValidatorSelectionSimple, SelectionError};
pub use zk_verifier::{ZKVerifier, VerificationResult, MockVerifier, ZKTrainingProof, VerificationMetadata};
pub use network::{ValidationMessage, ValidationNetwork, SignatureAggregator, NetworkEvent, NetworkError, ValidationSignature, AggregatedSignature};
pub use config::{ValidationParams, ConfigError as ValidationError};
