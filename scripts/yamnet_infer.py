#!/usr/bin/env python3
"""
yamnet_infer.py -- Google YAMNet inference for BirdCLEF acoustic classification.

Outputs JSON predictions compatible with the Xenom genetics-l2 worker.

Usage:
  python3 yamnet_infer.py --input <audio_file_or_dir> --output <output_dir> [--cpu] [--min_conf 0.1]

Dependencies:
  pip install tensorflow tensorflow-hub librosa numpy
"""

import argparse
import json
import os
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac"}
YAMNET_MODEL_URL = "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet"
SAMPLE_RATE = 16000  # YAMNet expects 16kHz


def main():
    parser = argparse.ArgumentParser(description="YAMNet bird audio classifier")
    parser.add_argument("--input",    required=True, help="Audio file or directory")
    parser.add_argument("--output",   required=True, help="Output directory for JSON")
    parser.add_argument("--min_conf", type=float, default=0.1, help="Minimum confidence threshold")
    parser.add_argument("--cpu",      action="store_true", help="Disable GPU (CPU-only inference)")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import numpy as np
        import librosa
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as e:
        _fatal(f"Missing dependency: {e}\n"
               "Run: pip install tensorflow tensorflow-hub librosa numpy")

    # Load YAMNet model from TensorFlow Hub
    try:
        print("Loading YAMNet model...", file=sys.stderr)
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet model loaded successfully", file=sys.stderr)
    except Exception as e:
        _fatal(f"Failed to load YAMNet model: {e}")

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

    all_detections = []

    for audio_file in audio_files:
        try:
            # Load audio at 16kHz (YAMNet's expected sample rate)
            waveform, _ = librosa.load(str(audio_file), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            all_detections.append({"file": str(audio_file), "error": str(e)})
            continue

        try:
            # YAMNet inference
            scores, embeddings, spectrogram = model(waveform)
            
            # scores shape: [num_frames, num_classes]
            # Get the class with highest score for each frame
            class_scores = scores.numpy()
            
            for frame_idx, frame_scores in enumerate(class_scores):
                top_idx = int(np.argmax(frame_scores))
                conf = float(frame_scores[top_idx])
                
                if conf >= args.min_conf:
                    # YAMNet processes 0.96s frames with 0.48s hop
                    start_s = frame_idx * 0.48
                    end_s = start_s + 0.96
                    
                    all_detections.append({
                        "file": str(audio_file),
                        "start_s": round(start_s, 2),
                        "end_s": round(end_s, 2),
                        "class_idx": top_idx,
                        "confidence": round(conf, 4),
                    })
        except Exception as e:
            all_detections.append({"file": str(audio_file), "error": str(e)})
            continue

    result = {"detections": all_detections}
    _output(args.output, input_path.stem, result)
    print(json.dumps(result))


def _output(output_dir: str, stem: str, data: dict):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.json").write_text(json.dumps(data, indent=2))


def _fatal(msg: str):
    print(json.dumps({"detections": [], "error": msg}))
    sys.exit(1)


if __name__ == "__main__":
    main()
