//! Gradient utilities shared across trainers (commitment, compression, averaging,
//! encryption for FedAvg updates).

use std::collections::HashMap;

use anyhow::{Context, Result};
use borsh::to_vec as borsh_to_vec;
use candle_core::{DType, Device, Tensor};
use rayon::prelude::*;

use crate::rpc::messages::{GradientLayer, GradientPayload, GradientUpdate};
use crate::trainer::mixed_precision::to_grad_dtype;
use crate::trainer::TrainingResult;

/// Compute a deterministic gradient commitment hash from a name -> tensor map.
pub(crate) fn gradient_commitment(grads: &HashMap<String, Tensor>) -> Result<[u8; 32]> {
    let mut hasher = blake3::Hasher::new();
    let mut names: Vec<_> = grads.keys().cloned().collect();
    names.sort();
    for name in names {
        let grad = &grads[&name];
        let grad_f32 = grad.to_dtype(DType::F32)?;
        let values = grad_f32.flatten_all()?.to_vec1::<f32>()?;
        hasher.update(name.as_bytes());
        for value in values {
            hasher.update(&value.to_le_bytes());
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Encrypt and package averaged gradients as a `GradientUpdate` for the seed-node.
pub(crate) fn build_gradient_update(
    model_id: &str,
    base_checkpoint: [u8; 32],
    named_grads: HashMap<String, Tensor>,
    participant_weight: f32,
    top_k_ratio: f32,
    result: &TrainingResult,
) -> Result<GradientUpdate> {
    let top_k_ratio = top_k_ratio.clamp(0.0, 1.0);

    // The commitment is over the plaintext gradients/weight-delta that the node
    // will receive after decryption. Compute it before the HashMap is consumed.
    let gradients_commitment = gradient_commitment(&named_grads)?;

    let mut pairs: Vec<(String, Tensor)> = named_grads.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let layer_entries: Vec<Result<(String, GradientLayer)>> = pairs
        .into_par_iter()
        .map(|(name, grad)| {
            let shape = grad.dims().to_vec();
            let flat = grad.flatten_all()?.to_vec1::<f32>()?;
            let (values, indices) = if top_k_ratio >= 1.0 { (flat, Vec::new()) } else { top_k_compress(&flat, top_k_ratio) };
            Ok((name, GradientLayer { values, shape, indices }))
        })
        .collect();

    let mut layer_gradients = HashMap::with_capacity(layer_entries.len());
    for entry in layer_entries {
        let (name, layer) = entry?;
        layer_gradients.insert(name, layer);
    }

    let payload = GradientPayload { layer_gradients };
    let payload_bytes = borsh_to_vec(&payload).context("Failed to serialize gradient payload")?;
    let encrypted_payload =
        model_crypto::encrypt(&payload_bytes, &model_crypto::derive_encryption_key()).context("Failed to encrypt gradient payload")?;

    Ok(GradientUpdate {
        model_id: model_id.to_string(),
        base_checkpoint,
        encrypted_payload,
        participant_weight,
        loss_before: result.loss_before,
        loss_after: result.loss_after,
        gradients_commitment,
        batch_indices: result.batch_indices.clone(),
        compute_time_ms: result.compute_time_ms,
    })
}

/// Keep only the `k` largest absolute values of `flat` and return them together
/// with their flattened indices (sorted ascending by index).
pub(crate) fn top_k_compress(flat: &[f32], ratio: f32) -> (Vec<f32>, Vec<usize>) {
    if ratio <= 0.0 || flat.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let k = ((flat.len() as f32 * ratio).ceil() as usize).clamp(1, flat.len());

    let mut indexed: Vec<(usize, f32)> = flat.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.abs().total_cmp(&a.1.abs()).then_with(|| b.0.cmp(&a.0)));
    let mut top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
    top.sort_by(|a, b| a.0.cmp(&b.0));

    let indices = top.iter().map(|(i, _)| *i).collect();
    let values = top.iter().map(|(_, v)| *v).collect();
    (values, indices)
}

/// Average a list of per-GPU gradient maps. All tensors are expected to be on the
/// CPU and in F32.
pub(crate) fn average_grad_maps(grads: &[HashMap<String, Tensor>]) -> Result<HashMap<String, Tensor>> {
    if grads.is_empty() {
        anyhow::bail!("Cannot average empty gradient list");
    }

    let mut acc: HashMap<String, (Tensor, usize)> = HashMap::new();
    for g in grads {
        for (name, t) in g.iter() {
            let t = to_grad_dtype(t)?;
            match acc.get_mut(name) {
                Some((sum, count)) => {
                    *sum = (&*sum + &t)?;
                    *count += 1;
                }
                None => {
                    acc.insert(name.clone(), (t, 1));
                }
            }
        }
    }

    let mut out = HashMap::with_capacity(acc.len());
    for (name, (sum, count)) in acc {
        let avg = (&sum / (count as f64))?;
        out.insert(name, avg);
    }

    Ok(out)
}

/// Element-wise addition of two gradient maps.
pub(crate) fn add_grad_maps(a: HashMap<String, Tensor>, b: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(a.len().max(b.len()));
    for (name, ta) in a {
        if let Some(tb) = b.get(&name) {
            let ta = to_grad_dtype(&ta)?;
            let tb = to_grad_dtype(tb)?;
            out.insert(name, (&ta + &tb)?);
        } else {
            out.insert(name, ta);
        }
    }
    for (name, tb) in b {
        out.entry(name).or_insert(tb);
    }
    Ok(out)
}

/// Move every tensor in a named gradient map to `device`.
pub(crate) fn move_grads_to_device(grads: HashMap<String, Tensor>, device: &Device) -> Result<HashMap<String, Tensor>> {
    grads.into_iter().map(|(k, v)| Ok((k, v.to_device(device)?))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_compress_keeps_largest_absolute_values() {
        let flat = vec![1.0f32, -5.0, 2.0, 0.1, -3.0, 4.0];
        let (values, indices) = top_k_compress(&flat, 0.5);
        assert_eq!(values.len(), 3);
        assert_eq!(indices.len(), 3);

        let mut reconstructed = vec![0.0f32; flat.len()];
        for (i, idx) in indices.iter().enumerate() {
            reconstructed[*idx] = values[i];
        }
        assert_eq!(reconstructed, vec![0.0, -5.0, 0.0, 0.0, -3.0, 4.0]);
    }

    #[test]
    fn test_top_k_compress_full_ratio_returns_sorted_identity() {
        let flat = vec![1.0f32, -5.0, 2.0, 0.1, -3.0, 4.0];
        let (values, indices) = top_k_compress(&flat, 1.0);
        assert_eq!(values.len(), flat.len());
        assert_eq!(indices, (0..flat.len()).collect::<Vec<_>>());
        assert_eq!(values, flat);
    }
}
