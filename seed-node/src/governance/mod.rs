//! On-chain governance client for ModelGovernance.sol.
//!
//! Seed nodes can propose new models, vote on proposals, and execute approved
//! proposals without requiring a hard fork.

pub mod proposer;
pub mod voter;

pub(crate) type ProposalRaw = (
    String,
    String,
    String,
    ethers::types::H256,
    ethers::types::U256,
    ethers::types::U256,
    ethers::types::U256,
    ethers::types::U256,
    ethers::types::U256,
    ethers::types::U256,
    ethers::types::U256,
    bool,
);
