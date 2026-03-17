#!/usr/bin/env python3
"""
yamnet_infer.py -- YAMNet inference for BirdCLEF 2026 acoustic classification.

Outputs:
  submission.csv  -- BirdCLEF 2026 format (row_id + 234 species columns, one row per 5-sec chunk)
  predictions.json -- {"score": float, "rows": int} for the Xenom miner

Usage:
  python3 yamnet_infer.py --input <audio_file_or_dir> --output <output_dir> [--cpu]

Dependencies:
  pip install tensorflow tensorflow-hub librosa numpy
"""

import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from birdclef2026 import SPECIES, N_SPECIES, UNIFORM_P, AUDIO_EXTS, collect_audio, write_submission, uniform_probs, load_expected_rows
SAMPLE_RATE = 16000   # YAMNet expects 16 kHz
CLIP_SECS   = 5
CLIP_LEN    = SAMPLE_RATE * CLIP_SECS

# YAMNet class indices that correspond to bird/animal sounds (approximate)
# Used to derive a "bird presence" probability per chunk
BIRD_CLASS_IDXS = list(range(80, 100))  # YAMNet classes ~80-99 include bird sounds



def yamnet_to_species_probs(frame_scores, np):
    """Map YAMNet 521-class scores to a 234-species probability vector.

    YAMNet is a general audio classifier, not a per-species bird classifier.
    Strategy: use the sum of bird-related class scores as a "bird presence"
    signal, then weight against a uniform prior using softmax over BIRD_CLASS_IDXS.
    """
    bird_scores = frame_scores[BIRD_CLASS_IDXS]
    # softmax over bird classes → relative weights for bird sounds
    bird_exp   = np.exp(bird_scores - bird_scores.max())
    bird_probs = bird_exp / bird_exp.sum()  # sums to 1 over ~20 bird classes

    # Map 20 bird classes → 234 species (cyclic assignment)
    species_probs = np.full(N_SPECIES, UNIFORM_P * 0.5)
    for i, p in enumerate(bird_probs):
        idx = i % N_SPECIES
        species_probs[idx] += p / (N_SPECIES // len(bird_probs) + 1)

    # Normalise to sum = 1
    total = species_probs.sum()
    if total > 0:
        species_probs /= total
    return species_probs


def main():
    parser = argparse.ArgumentParser(description="YAMNet BirdCLEF 2026 inference")
    parser.add_argument("--input",  required=True, help="Audio file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cpu",    action="store_true", help="Disable GPU")
    parser.add_argument("--sample-submission", dest="sample_submission",
                        help="Path to sample_submission.csv — output will match its row_ids exactly")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import numpy as np
        import librosa
        import tensorflow_hub as hub
    except ImportError as e:
        _fatal(f"Missing dependency: {e}\nRun: pip install tensorflow tensorflow-hub librosa numpy")

    try:
        print("[yamnet] Loading model...", file=sys.stderr)
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("[yamnet] Model ready", file=sys.stderr)
    except Exception as e:
        _fatal(f"Failed to load YAMNet: {e}")

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build stem → Path map
    audio_map: dict = {}
    for f in collect_audio(input_path):
        audio_map[f.stem] = f
    for subdir in ("test_soundscapes", "train_soundscapes"):
        sub = input_path / subdir
        if sub.is_dir():
            for f in collect_audio(sub):
                audio_map.setdefault(f.stem, f)

    if args.sample_submission and Path(args.sample_submission).exists():
        expected = load_expected_rows(args.sample_submission)
        print(f"[yamnet] sample_submission: {len(expected)} soundscapes, "
              f"{sum(len(v) for v in expected.values())} rows", file=sys.stderr)
    else:
        expected = {stem: [(i + 1) * CLIP_SECS for i in range(
            max(1, len(librosa.load(str(p), sr=SAMPLE_RATE, mono=True, duration=None)[0]) // CLIP_LEN)
        )] for stem, p in audio_map.items()}

    if not expected:
        write_submission(args.output, [], 0.0)
        print(json.dumps({"score": 0.0, "rows": 0}))
        return

    audio_cache: dict = {}
    rows:      list   = []
    max_probs: list   = []

    for stem, end_secs in expected.items():
        audio_file = audio_map.get(stem)

        if audio_file and stem not in audio_cache:
            try:
                waveform, _ = librosa.load(str(audio_file), sr=SAMPLE_RATE, mono=True)
                audio_cache[stem] = waveform
            except Exception as e:
                print(f"[yamnet] skip {audio_file.name}: {e}", file=sys.stderr)

        waveform = audio_cache.get(stem)

        for end_sec in end_secs:
            row_id = f"{stem}_{end_sec}"
            if waveform is not None:
                start = max(0, (end_sec - CLIP_SECS) * SAMPLE_RATE)
                clip  = waveform[start: start + CLIP_LEN]
                if len(clip) < CLIP_LEN:
                    clip = np.pad(clip, (0, CLIP_LEN - len(clip)))
                try:
                    scores, _, _ = model(clip)
                    frame_scores = scores.numpy().mean(axis=0)
                    probs = yamnet_to_species_probs(frame_scores, np)
                except Exception as e:
                    print(f"[yamnet] {row_id} error: {e}", file=sys.stderr)
                    probs = np.full(N_SPECIES, UNIFORM_P)
            else:
                probs = np.full(N_SPECIES, UNIFORM_P)

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
