#!/usr/bin/env python3
"""
efficientnet_infer.py -- EfficientNet-B0 BirdCLEF 2026 inference.
Based on: https://www.kaggle.com/models/haradibots/bird-efficientnet-model/PyTorch/default

Outputs:
  submission.csv  -- BirdCLEF 2026 format (row_id + 234 species, one row per 5-sec chunk)
  predictions.json -- {"score": float, "rows": int}

Usage:
  python3 efficientnet_infer.py --input <audio_dir> --output <output_dir> [--cpu]

Dependencies:
  pip install torch torchaudio timm numpy kaggle
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from birdclef2026 import SPECIES, N_SPECIES, AUDIO_EXTS, collect_audio, write_submission, uniform_probs, load_expected_rows

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_CACHE  = Path.home() / ".cache" / "xenom" / "bird-efficientnet"
KAGGLE_MODEL = "haradibots/bird-efficientnet-model/PyTorch/default/1"
SR           = 32000
DURATION     = 5
SAMPLES      = SR * DURATION

# The model was trained with labels = sorted(train.csv primary_label.unique()).
# We approximate this with sorted(SPECIES) — same set, alphabetical order.
MODEL_LABELS = sorted(SPECIES)
# Map model output index → submission column index
MODEL_TO_SUBMISSION = [SPECIES.index(lbl) for lbl in MODEL_LABELS]

# Mel spectrogram transform (matches Kaggle notebook exactly)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
)


# ── Audio → mel tensor ────────────────────────────────────────────────────────

def audio_to_mel(clip: torch.Tensor) -> torch.Tensor:
    """clip: [samples]  →  [1, 224, 224]"""
    mel = mel_transform(clip)                    # [128, T]
    mel = torch.log(mel + 1e-6)                  # log-mel
    mel = mel.unsqueeze(0)                        # [1, 128, T]
    mel = torch.nn.functional.interpolate(
        mel.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)                                  # [1, 224, 224]
    return mel


# ── Model ─────────────────────────────────────────────────────────────────────

class BirdModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        import timm
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            in_chans=1,
        )
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ── Model download / cache ────────────────────────────────────────────────────

def ensure_model() -> Path:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    ready = MODEL_CACHE / ".ready"
    if ready.exists():
        w = _find_weights(MODEL_CACHE)
        if w:
            return w

    print("[efficientnet] Downloading from Kaggle...", file=sys.stderr)
    try:
        subprocess.run(
            ["kaggle", "models", "instances", "versions", "download",
             KAGGLE_MODEL, "--path", str(MODEL_CACHE)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"kaggle download failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("kaggle CLI not found — pip install kaggle")

    for arc in list(MODEL_CACHE.glob("*.tar.gz")) + list(MODEL_CACHE.glob("*.tgz")):
        with tarfile.open(arc) as t:
            t.extractall(MODEL_CACHE)
        arc.unlink()
    for arc in MODEL_CACHE.glob("*.zip"):
        with zipfile.ZipFile(arc) as z:
            z.extractall(MODEL_CACHE)
        arc.unlink()

    w = _find_weights(MODEL_CACHE)
    if not w:
        raise RuntimeError(f"No .pth/.pt found in {MODEL_CACHE}")
    ready.touch()
    print(f"[efficientnet] Model ready: {w}", file=sys.stderr)
    return w


def _find_weights(base: Path) -> "Path | None":
    for pat in ("*.pth", "*.pt"):
        hits = sorted(base.rglob(pat))
        if hits:
            return hits[0]
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EfficientNet BirdCLEF 2026 inference")
    parser.add_argument("--input",  required=True, help="Audio file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cpu",    action="store_true", help="Force CPU")
    parser.add_argument("--sample-submission", dest="sample_submission",
                        help="Path to sample_submission.csv — output will match its row_ids exactly")
    args = parser.parse_args()

    # Device selection
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[efficientnet] device={device}", file=sys.stderr)

    # Load model
    try:
        weights_path = ensure_model()
    except Exception as e:
        _fatal(f"Model download failed: {e}")

    print(f"[efficientnet] Loading {weights_path}", file=sys.stderr)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    num_classes = _infer_num_classes(state_dict)
    print(f"[efficientnet] num_classes={num_classes}", file=sys.stderr)

    model = BirdModel(num_classes).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    mel_transform.to(device)

    input_path = Path(args.input)

    # Build stem → Path map for quick audio lookup
    audio_map: dict = {}
    for f in collect_audio(input_path):
        audio_map[f.stem] = f
    # Also search subdirs explicitly (test_soundscapes/, train_soundscapes/)
    for subdir in ("test_soundscapes", "train_soundscapes"):
        sub = input_path / subdir
        if sub.is_dir():
            for f in collect_audio(sub):
                audio_map.setdefault(f.stem, f)

    # Determine iteration order
    if args.sample_submission and Path(args.sample_submission).exists():
        expected = load_expected_rows(args.sample_submission)
        print(f"[efficientnet] sample_submission: {len(expected)} soundscapes, "
              f"{sum(len(v) for v in expected.values())} rows", file=sys.stderr)
    else:
        # Build expected from discovered audio files (original behaviour)
        expected = {}
        for f in collect_audio(input_path):
            total = max(1, _audio_duration_clips(f))
            expected[f.stem] = [(i + 1) * DURATION for i in range(total)]

    if not expected:
        write_submission(args.output, [], 0.0)
        print(json.dumps({"score": 0.0, "rows": 0}))
        return

    audio_cache: dict = {}
    rows:      list   = []
    max_probs: list   = []

    for stem, end_secs in expected.items():
        audio_file = audio_map.get(stem)

        if audio_file and str(audio_file) not in audio_cache:
            try:
                waveform, sr = torchaudio.load(str(audio_file))
                if sr != SR:
                    waveform = torchaudio.functional.resample(waveform, sr, SR)
                audio_cache[str(audio_file)] = waveform[0]  # mono
            except Exception as e:
                print(f"[efficientnet] skip {audio_file.name}: {e}", file=sys.stderr)
                audio_file = None

        audio = audio_cache.get(str(audio_file)) if audio_file else None

        for end_sec in end_secs:
            row_id = f"{stem}_{end_sec}"

            if audio is not None:
                start = max(0, (end_sec - DURATION) * SR)
                clip  = audio[start: start + SAMPLES]
                if clip.shape[0] < SAMPLES:
                    clip = torch.nn.functional.pad(clip, (0, SAMPLES - clip.shape[0]))
                try:
                    mel    = audio_to_mel(clip.to(device))
                    tensor = mel.unsqueeze(0)
                    with torch.no_grad():
                        out  = model(tensor)
                        prob = torch.sigmoid(out).cpu().numpy()[0]
                    species_probs = uniform_probs(np) * 1e-6
                    for model_idx, sub_idx in enumerate(MODEL_TO_SUBMISSION):
                        if model_idx < len(prob):
                            species_probs[sub_idx] = float(prob[model_idx])
                except Exception as e:
                    print(f"[efficientnet] {row_id} error: {e}", file=sys.stderr)
                    species_probs = uniform_probs(np)
            else:
                species_probs = uniform_probs(np)

            max_probs.append(float(species_probs.max()))
            rows.append((row_id, species_probs))

    score = float(np.mean(max_probs)) if max_probs else 0.0
    write_submission(args.output, rows, score)
    print(json.dumps({"score": score, "rows": len(rows)}))


def _audio_duration_clips(audio_file: Path) -> int:
    """Return number of 5-second clips in an audio file without fully loading it."""
    try:
        info = torchaudio.info(str(audio_file))
        return max(1, info.num_frames // (SR * DURATION))
    except Exception:
        return 1


def _infer_num_classes(state_dict: dict) -> int:
    for key in reversed(list(state_dict.keys())):
        if "classifier" in key and "weight" in key:
            return state_dict[key].shape[0]
    return N_SPECIES


def _fatal(msg: str):
    print(json.dumps({"score": 0.0, "error": msg}))
    sys.exit(1)


if __name__ == "__main__":
    main()
