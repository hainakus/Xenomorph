//! Unified training coordinator.
//!
//! This module merges the seed-node's responsibilities (model download,
//! genome archive serving, miner WebSocket RPC) into the full node. Miners
//! connect directly to `xenom` via WebSocket, download model checkpoints
//! and genome batches, and submit training blocks that the node validates
//! and mines into the DAG.

pub mod coordinator;
pub mod service;
pub mod websocket_server;
