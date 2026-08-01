//! Shared Borsh RPC messages for the Xenomorph miner/orchestrator protocol.
//!
//! This crate is used by both `xenom-miner` and `seed-node` (and the unified
//! `xenom` node) so all parties share the same wire format without duplication.

pub mod messages;
