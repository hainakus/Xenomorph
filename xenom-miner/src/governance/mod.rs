//! Miner participation in on-chain governance.

pub mod voter;

pub use voter::{CastVoteRequest, ExecuteRequest, GovernanceVoter, ProposalSummary};
