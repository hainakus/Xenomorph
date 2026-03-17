#!/usr/bin/env python3
"""
perch_infer.py -- Google Perch v2 inference for BirdCLEF 2026.

Outputs:
  submission.csv  -- BirdCLEF 2026 format (row_id + 234 species, one row per 5-sec chunk)
  predictions.json -- {"score": float, "rows": int}

Perch outputs per-XC-ID probabilities mapped directly to the 234 BirdCLEF 2026 species.

Usage:
  python3 perch_infer.py --input <audio_file_or_dir> --output <output_dir> [--cpu]

Dependencies:
  pip install kagglehub tensorflow librosa numpy
"""

import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from birdclef2026 import SPECIES, N_SPECIES, UNIFORM_P, NUMERIC_XC_IDS, collect_audio, write_submission, uniform_probs

AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac"}
MODEL_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/perch-v2"
SAMPLE_RATE  = 32000
CLIP_SECS    = 5
CLIP_LEN     = SAMPLE_RATE * CLIP_SECS


def perch_probs_to_species(raw_probs, np, perch_class_list):
    """Map Perch output (10k+ XC species) to the 234 BirdCLEF 2026 species vector."""
    species_probs = uniform_probs(np) * 0.01  # tiny uniform prior
    if perch_class_list is not None:
        for i, xc_id in enumerate(perch_class_list):
            xc_str = str(xc_id)
            if xc_str in NUMERIC_XC_IDS and i < len(raw_probs):
                idx = SPECIES.index(xc_str)
                species_probs[idx] = float(raw_probs[i])
    total = species_probs.sum()
    if total > 0:
        species_probs /= total
    return species_probs


def main():
    parser = argparse.ArgumentParser(description="Perch v2 BirdCLEF 2026 inference")
    parser.add_argument("--input",  required=True, help="Audio file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cpu",    action="store_true", help="Disable GPU")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import kagglehub
        import numpy as np
        import librosa
        import tensorflow as tf
    except ImportError as e:
        _fatal(f"Missing dependency: {e}\nRun: pip install kagglehub tensorflow librosa numpy")

    try:
        print("[perch] Downloading model...", file=sys.stderr)
        model_path = kagglehub.model_download(MODEL_HANDLE)
        model = tf.saved_model.load(model_path)
        infer = model.signatures.get("serving_default") or getattr(model, "infer_tf", None)
        perch_class_list = None
        if hasattr(model, "class_names"):
            perch_class_list = list(model.class_names.numpy())
        print("[perch] Model ready", file=sys.stderr)
    except Exception as e:
        _fatal(f"Failed to load Perch model: {e}")

    input_path = Path(args.input)
    audio_files = collect_audio(input_path)
    if not audio_files:
        write_submission(args.output, [], 0.0)
        print(json.dumps({"score": 0.0, "rows": 0}))
        return

    rows = []
    max_probs = []

    for audio_file in audio_files:
        stem = audio_file.stem
        try:
            waveform, _ = librosa.load(str(audio_file), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"[perch] skip {audio_file.name}: {e}", file=sys.stderr)
            continue

        n_clips = max(1, len(waveform) // CLIP_LEN)
        for clip_idx in range(n_clips):
            start = clip_idx * CLIP_LEN
            clip  = waveform[start: start + CLIP_LEN]
            if len(clip) < CLIP_LEN:
                clip = np.pad(clip, (0, CLIP_LEN - len(clip)))

            end_sec = (clip_idx + 1) * CLIP_SECS
            row_id  = f"{stem}_{end_sec}"

            clip_tensor = tf.constant(clip[np.newaxis, :], dtype=tf.float32)
            try:
                try:
                    result = model.infer_tf(clip_tensor)
                    logits = result[0] if isinstance(result, (list, tuple)) else result["output_0"]
                except Exception:
                    out    = infer(clip_tensor)
                    logits = list(out.values())[0]
                raw_probs = tf.nn.softmax(logits).numpy()[0]
                probs = perch_probs_to_species(raw_probs, np, perch_class_list)
            except Exception as e:
                print(f"[perch] inference error clip {clip_idx}: {e}", file=sys.stderr)
                probs = uniform_probs(np)

            max_probs.append(float(probs.max()))
            rows.append((row_id, probs))

    score = float(np.mean(max_probs)) if max_probs else 0.0
    write_submission(args.output, rows, score)
    print(json.dumps({"score": score, "rows": len(rows)}))


def _fatal(msg: str):
    print(json.dumps({"score": 0.0, "error": msg}))
    sys.exit(1)


if __name__ == "__main__":
    main()
