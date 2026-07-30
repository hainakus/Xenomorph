//! Gradient utilities shared across trainers (commitment, compression, averaging,
//! encryption for FedAvg updates).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use borsh::{from_slice as borsh_from_slice, to_vec as borsh_to_vec};
use candle_core::{DType, Device, Tensor};
use rayon::prelude::*;

use crate::rpc::messages::{GenomeSlice, GradientLayer, GradientPayload, GradientUpdate};
use crate::trainer::mixed_precision::to_grad_dtype;
use crate::trainer::TrainingResult;

/// Compute a deterministic gradient commitment hash from a name -> tensor map.
pub fn gradient_commitment(grads: &HashMap<String, Tensor>) -> Result<[u8; 32]> {
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
    batch_id: u64,
    learning_rate: f32,
    genome_merkle_root: [u8; 32],
    genome_slices: Vec<GenomeSlice>,
) -> Result<GradientUpdate> {
    let top_k_ratio = top_k_ratio.clamp(0.0, 1.0);

    let mut pairs: Vec<(String, Tensor)> = named_grads.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // Metal uses a single command queue; concurrent readbacks from multiple
    // threads can return `WouldBlock`. Fall back to sequential iteration when
    // all gradients live on a Metal device. CPU/CUDA still benefit from Rayon.
    let use_par = !pairs.iter().all(|(_, grad)| grad.device().is_metal());

    let layer_entries: Vec<Result<(String, GradientLayer)>> = if use_par {
        pairs
            .into_par_iter()
            .map(|(name, grad)| {
                let shape = grad.dims().to_vec();
                let flat = grad.flatten_all()?.to_vec1::<f32>()?;
                let (values, indices) = if top_k_ratio >= 1.0 { (flat, Vec::new()) } else { top_k_compress(&flat, top_k_ratio) };
                Ok((name, GradientLayer { values, shape, indices }))
            })
            .collect()
    } else {
        pairs
            .into_iter()
            .map(|(name, grad)| {
                let shape = grad.dims().to_vec();
                let flat = grad.flatten_all()?.to_vec1::<f32>()?;
                let (values, indices) = if top_k_ratio >= 1.0 { (flat, Vec::new()) } else { top_k_compress(&flat, top_k_ratio) };
                Ok((name, GradientLayer { values, shape, indices }))
            })
            .collect()
    };

    let mut layer_gradients = HashMap::with_capacity(layer_entries.len());
    for entry in layer_entries {
        let (name, layer) = entry?;
        layer_gradients.insert(name, layer);
    }

    // The commitment must be computed over the exact plaintext the node will
    // reconstruct after decryption. If top-k compression is enabled, the dropped
    // entries become zeros; reconstruct that tensor map and hash it.
    let reconstructed = reconstruct_from_layers(&layer_gradients)?;
    let gradients_commitment = gradient_commitment(&reconstructed)?;

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
        batch_id,
        learning_rate,
        genome_merkle_root,
        genome_slices,
        compute_time_ms: result.compute_time_ms,
    })
}

/// Decrypt and reconstruct the weight-space delta contained in a `GradientUpdate`.
/// This is used by validators and seed-nodes to verify the commitment and to apply
/// the update without having to re-run local training.
pub fn reconstruct_gradient_update(update: &GradientUpdate) -> Result<HashMap<String, Tensor>> {
    let payload_bytes = model_crypto::decrypt(&update.encrypted_payload, &model_crypto::derive_encryption_key())
        .context("Failed to decrypt gradient payload")?;
    let payload: GradientPayload = borsh_from_slice(&payload_bytes).context("Failed to deserialize gradient payload")?;
    reconstruct_from_layers(&payload.layer_gradients)
}

/// Reconstruct a dense `HashMap<String, Tensor>` from compressed `GradientLayer`s
/// the same way the seed-node will after decrypting the payload.
pub fn reconstruct_from_layers(layers: &HashMap<String, GradientLayer>) -> Result<HashMap<String, Tensor>> {
    let device = Device::Cpu;
    let mut reconstructed = HashMap::with_capacity(layers.len());
    for (name, layer) in layers {
        let total_len: usize = layer.shape.iter().product();
        if layer.indices.is_empty() {
            let tensor = Tensor::from_vec(layer.values.clone(), layer.shape.clone(), &device)?;
            reconstructed.insert(name.clone(), tensor);
            continue;
        }

        if layer.values.len() != layer.indices.len() {
            bail!("Compressed gradient has {} values but {} indices", layer.values.len(), layer.indices.len());
        }
        let mut flat = vec![0.0f32; total_len];
        for (idx, value) in layer.indices.iter().zip(layer.values.iter()) {
            if *idx >= total_len {
                bail!("Gradient index {} out of bounds for shape {:?}", idx, layer.shape);
            }
            flat[*idx] = *value;
        }
        let tensor = Tensor::from_vec(flat, layer.shape.clone(), &device)?;
        reconstructed.insert(name.clone(), tensor);
    }
    Ok(reconstructed)
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

/// Clip a gradient map by its global L2 norm. Returns a new map of F32 tensors.
pub(crate) fn clip_grad_norm(named_grads: &HashMap<String, Tensor>, max_norm: f64) -> Result<HashMap<String, Tensor>> {
    let mut total_sq = 0.0f64;
    for t in named_grads.values() {
        let t_f32 = t.to_dtype(DType::F32)?;
        total_sq += t_f32.sqr()?.sum_all()?.to_vec0::<f32>()? as f64;
    }
    let norm = total_sq.sqrt();
    if norm <= max_norm || norm == 0.0 {
        let mut out = HashMap::with_capacity(named_grads.len());
        for (k, v) in named_grads {
            out.insert(k.clone(), v.to_dtype(DType::F32)?.copy()?);
        }
        Ok(out)
    } else {
        scale_grad_map(named_grads.clone(), max_norm / norm)
    }
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

/// Scale every tensor in a named gradient map by a scalar factor.
pub(crate) fn scale_grad_map(grads: HashMap<String, Tensor>, scale: f64) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(grads.len());
    for (name, t) in grads {
        let t = to_grad_dtype(&t)?;
        out.insert(name, (&t * scale)?);
    }
    Ok(out)
}

/// Sum a list of named gradient maps, ignoring missing entries. The caller is
/// expected to divide by the total weight afterwards when computing a weighted
/// average.
pub(crate) fn sum_grad_maps(grads: &[HashMap<String, Tensor>]) -> Result<HashMap<String, Tensor>> {
    if grads.is_empty() {
        anyhow::bail!("Cannot sum empty gradient list");
    }

    let mut acc: HashMap<String, Tensor> = HashMap::new();
    for g in grads {
        for (name, t) in g.iter() {
            let t = to_grad_dtype(t)?;
            match acc.get_mut(name) {
                Some(sum) => {
                    *sum = (&*sum + &t)?;
                }
                None => {
                    acc.insert(name.clone(), t);
                }
            }
        }
    }
    Ok(acc)
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
