use serde_json::Value;
use std::sync::Arc;

use crate::job::Job;
use crate::l2_jobs::{L2Job, L2JobSlot};

// ── Themed dispatcher ─────────────────────────────────────────────────────────

/// Packages a PoW job and an optional L2 job into the fields needed for
/// `mining.notify` param[6].
///
/// The `l2_job` field is `null` when no L2 pool is configured or no job is
/// available.  Miners that do not support L2 simply ignore param[6].
pub struct DispatchedJob<'a> {
    #[allow(dead_code)]
    pub pow:    &'a Job,
    pub l2_val: Value,   // serde_json::Value::Null or L2Job object
}

/// Build a `DispatchedJob` by reading the current L2 slot (non-blocking).
///
/// Returns immediately with `l2_val = null` if no L2 job is available or the
/// slot lock is contended.
pub async fn dispatch<'a>(pow: &'a Job, slot: &L2JobSlot) -> DispatchedJob<'a> {
    let l2_val = match slot.try_read() {
        Ok(guard) => guard
            .as_ref()
            .map(|j| j.to_value())
            .unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    DispatchedJob { pow, l2_val }
}

/// Blocking variant — awaits the read lock.
#[allow(dead_code)]
pub async fn dispatch_await<'a>(pow: &'a Job, slot: &L2JobSlot) -> DispatchedJob<'a> {
    let l2_val = slot
        .read()
        .await
        .as_ref()
        .map(|j: &Arc<L2Job>| j.to_value())
        .unwrap_or(Value::Null);
    DispatchedJob { pow, l2_val }
}
