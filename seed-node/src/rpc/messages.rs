//! Re-export of shared RPC messages from the `xenom-rpc` crate.
//!
//! New code should prefer `use xenom_rpc::messages::*` directly.  This module
//! exists as a compatibility shim while the rest of the codebase is migrated.

pub use xenom_rpc::messages::*;
