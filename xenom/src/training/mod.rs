//! Unified training and inference coordinator.
//!
//! This module merges the seed-node's responsibilities (model download,
//! genome archive serving, miner WebSocket RPC, and gRPC inference) into the
//! full node. Miners connect directly to `xenom` via WebSocket, download model
//! checkpoints and genome batches, and submit training blocks that the node
//! validates and mines into the DAG. API gateways can also connect directly to
//! `xenom` for OpenAI-compatible inference.

pub mod checkpoint_sync;
pub mod coordinator;
pub mod gradient_validator;
pub mod inference_service;
pub mod service;
pub mod websocket_server;
