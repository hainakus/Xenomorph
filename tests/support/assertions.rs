use anyhow::{Context, Result};

pub fn assert_ok<T>(result: Result<T>, message: &str) {
    assert!(result.is_ok(), "{}: {:?}", message, result.err());
}

pub fn assert_err_contains<T>(result: Result<T>, needle: &str) {
    let err = result.err().with_context(|| format!("expected an error containing '{}'", needle)).unwrap().to_string();
    assert!(err.to_lowercase().contains(&needle.to_lowercase()), "error does not contain '{}': {}", needle, err);
}

pub fn assert_loss_improved(before: f64, after: f64, min_improvement: f64) {
    let improvement = before - after;
    assert!(improvement >= min_improvement, "expected loss improvement >= {}, got {}", min_improvement, improvement);
}

pub async fn assert_balance_increased(initial: u64, final_: u64, expected_min: u64) {
    assert!(final_ > initial, "balance did not increase: {} -> {}", initial, final_);
    let reward = final_ - initial;
    assert!(reward >= expected_min, "reward {} is less than expected {}", reward, expected_min);
}
