#!/usr/bin/env python3
"""
perch_infer.py -- Google Perch v2 inference for BirdCLEF acoustic classification.

Outputs JSON predictions compatible with the Xenom genetics-l2 worker.

Usage:
  python3 perch_infer.py --input <audio_file_or_dir> --output <output_dir> [--cpu] [--min_conf 0.1]

Dependencies:
  pip install kagglehub tensorflow librosa numpy
  # GPU: pip install tensorflow[and-cuda]
"""

import argparse
import json
import os
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac"}
MODEL_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/perch-v2"
SAMPLE_RATE  = 32000
CLIP_SECS    = 5


def main():
    parser = argparse.ArgumentParser(description="Perch v2 bird audio classifier")
    parser.add_argument("--input",    required=True, help="Audio file or directory")
    parser.add_argument("--output",   required=True, help="Output directory for JSON")
    parser.add_argument("--min_conf", type=float, default=0.1, help="Minimum confidence threshold")
    parser.add_argument("--cpu",      action="store_true", help="Disable GPU (CPU-only inference)")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import kagglehub
        import numpy as np
        import librosa
        import tensorflow as tf
    except ImportError as e:
        _fatal(f"Missing dependency: {e}\n"
               "Run: pip install kagglehub tensorflow librosa numpy")

    # Load Perch v2 model (cached after first download)
    try:
        model_path = kagglehub.model_download(MODEL_HANDLE)
        model = tf.saved_model.load(model_path)
        infer = model.signatures.get("serving_default") or model.infer_tf
    except Exception as e:
        _fatal(f"Failed to load Perch model '{MODEL_HANDLE}': {e}")

    # Collect audio files
    input_path = Path(args.input)
    audio_files = []
    if input_path.is_file() and input_path.suffix.lower() in AUDIO_EXTS:
        audio_files = [input_path]
    elif input_path.is_dir():
        for p in sorted(input_path.rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS:
                audio_files.append(p)

    if not audio_files:
        _output(args.output, input_path.stem, {"detections": [], "note": "no audio files found"})
        return

    clip_len  = SAMPLE_RATE * CLIP_SECS
    all_detections = []

    for audio_file in audio_files:
        try:
            waveform, _ = librosa.load(str(audio_file), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            all_detections.append({"file": str(audio_file), "error": str(e)})
            continue

        clips = [waveform[i:i + clip_len] for i in range(0, len(waveform), clip_len)]
        for clip_idx, clip in enumerate(clips):
            if len(clip) < clip_len:
                clip = np.pad(clip, (0, clip_len - len(clip)))

            clip_tensor = tf.constant(clip[np.newaxis, :], dtype=tf.float32)
            try:
                result  = model.infer_tf(clip_tensor)
                logits  = result[0] if isinstance(result, (list, tuple)) else result["output_0"]
                probs   = tf.nn.softmax(logits).numpy()[0]
            except Exception:
                try:
                    out    = infer(clip_tensor)
                    logits = list(out.values())[0]
                    probs  = tf.nn.softmax(logits).numpy()[0]
                except Exception as e2:
                    all_detections.append({"file": str(audio_file), "clip": clip_idx, "error": str(e2)})
                    continue

            top_idx  = int(np.argmax(probs))
            conf     = float(probs[top_idx])
            if conf >= args.min_conf:
                all_detections.append({
                    "file":       str(audio_file),
                    "start_s":    clip_idx * CLIP_SECS,
                    "end_s":      (clip_idx + 1) * CLIP_SECS,
                    "species_idx": top_idx,
                    "confidence": round(conf, 4),
                })

    result = {"detections": all_detections}
    _output(args.output, input_path.stem, result)
    print(json.dumps(result))


def _output(output_dir: str, stem: str, data: dict):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}_perch.json").write_text(json.dumps(data, indent=2))


def _fatal(msg: str):
    print(json.dumps({"detections": [], "error": msg}))
    sys.exit(1)


if __name__ == "__main__":
    main()
