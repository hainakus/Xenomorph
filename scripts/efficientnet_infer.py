#!/usr/bin/env python3
# ruff: noqa
"""
efficientnet_infer.py -- PyTorch EfficientNet bird classifier for BirdCLEF.
Uses: https://www.kaggle.com/models/haradibots/bird-efficientnet-model/PyTorch/default

Compatible with macOS (CPU / Apple Silicon MPS).

Usage:
  python3 efficientnet_infer.py --input <audio_file_or_dir> --output <output_dir> [--cpu] [--min_conf 0.1]

Dependencies:
  pip install torch torchvision torchaudio librosa numpy kaggle
"""

import argparse
import json
import os
import sys
import subprocess
import tarfile
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from birdclef2026 import SPECIES, N_SPECIES, collect_audio, write_submission, uniform_probs

AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac"}
MODEL_CACHE = Path.home() / ".cache" / "xenom" / "bird-efficientnet"
KAGGLE_MODEL = "haradibots/bird-efficientnet-model/PyTorch/default/1"
SAMPLE_RATE  = 32000   # BirdCLEF standard
N_MELS       = 128
IMAGE_SIZE   = 224


# ── Model download / cache ────────────────────────────────────────────────────

def ensure_model() -> Path:
    """Download model from Kaggle if not cached. Returns path to weights file."""
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    ready_marker = MODEL_CACHE / ".ready"

    if ready_marker.exists():
        weights = _find_weights(MODEL_CACHE)
        if weights:
            return weights

    print("[efficientnet] Downloading model from Kaggle...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["kaggle", "models", "instances", "versions", "download",
             KAGGLE_MODEL, "--path", str(MODEL_CACHE)],
            capture_output=True, text=True, check=True
        )
        print(result.stdout, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"kaggle download failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("kaggle CLI not found. Install: pip install kaggle")

    # Extract archives
    for arc in list(MODEL_CACHE.glob("*.tar.gz")) + list(MODEL_CACHE.glob("*.tgz")):
        print(f"[efficientnet] Extracting {arc.name}...", file=sys.stderr)
        with tarfile.open(arc) as t:
            t.extractall(MODEL_CACHE)
        arc.unlink()

    for arc in MODEL_CACHE.glob("*.zip"):
        print(f"[efficientnet] Extracting {arc.name}...", file=sys.stderr)
        with zipfile.ZipFile(arc) as z:
            z.extractall(MODEL_CACHE)
        arc.unlink()

    weights = _find_weights(MODEL_CACHE)
    if not weights:
        raise RuntimeError(f"No model weights (.pth/.pt) found in {MODEL_CACHE}")

    ready_marker.touch()
    print(f"[efficientnet] Model ready: {weights}", file=sys.stderr)
    return weights


def _find_weights(base: Path) -> Path | None:
    for ext in ("*.pth", "*.pt"):
        hits = sorted(base.rglob(ext))
        if hits:
            return hits[0]
    return None


def _find_labels(base: Path) -> list[str] | None:
    for name in ("labels.txt", "class_names.txt", "classes.txt",
                 "labels.json", "class_names.json"):
        hits = list(base.rglob(name))
        if hits:
            f = hits[0]
            text = f.read_text().strip()
            if f.suffix == ".json":
                data = json.loads(text)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [v for _, v in sorted(data.items(), key=lambda x: int(x[0]))]
            else:
                return [l.strip() for l in text.splitlines() if l.strip()]
    return None


# ── Audio → mel spectrogram ───────────────────────────────────────────────────

def audio_to_tensor(audio_path: Path, device):
    """Load audio and return [1, 3, H, W] tensor scaled to IMAGE_SIZE."""
    import torch
    import numpy as np
    import librosa
    from torchvision import transforms

    waveform, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=5.0)

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=N_MELS, fmax=16000,
        n_fft=1024, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 255] uint8
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    img = (mel_db * 255).astype(np.uint8)

    # [H, W] → [1, H, W] → repeat to [3, H, W]
    t = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return preprocess(t / 255.0).unsqueeze(0).to(device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EfficientNet BirdCLEF 2026 inference")
    parser.add_argument("--input",  required=True, help="Audio file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cpu",    action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError as e:
        _fatal(f"Missing dependency: {e}\n"
               "Run: pip install torch torchvision torchaudio librosa")

    # Device: CUDA > MPS (Apple Silicon) > CPU
    if args.cpu or not torch.cuda.is_available():
        if not args.cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[efficientnet] Using Apple Silicon MPS", file=sys.stderr)
        else:
            device = torch.device("cpu")
            print("[efficientnet] Using CPU", file=sys.stderr)
    else:
        device = torch.device("cuda")
        print("[efficientnet] Using CUDA", file=sys.stderr)

    # Load model weights
    try:
        weights_path = ensure_model()
    except Exception as e:
        _fatal(f"Model download failed: {e}")

    print(f"[efficientnet] Loading weights: {weights_path}", file=sys.stderr)
    checkpoint = torch.load(weights_path, map_location=device)

    # Determine number of classes from checkpoint
    state_dict = checkpoint if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint \
        else checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    # Find classifier output size
    num_classes = _infer_num_classes(state_dict)
    print(f"[efficientnet] Classes: {num_classes}", file=sys.stderr)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    import torch.nn.functional as F
    import numpy as np

    labels = _find_labels(MODEL_CACHE) or [str(i) for i in range(num_classes)]
    # Build label→SPECIES index map
    label_to_species_idx = {}
    for i, lbl in enumerate(labels):
        if lbl in SPECIES:
            label_to_species_idx[i] = SPECIES.index(lbl)

    input_path = Path(args.input)
    audio_files = collect_audio(input_path)
    if not audio_files:
        write_submission(args.output, [], 0.0)
        print(json.dumps({"score": 0.0, "rows": 0}))
        return

    rows = []
    max_probs = []
    CLIP_SECS = 5

    for audio_file in audio_files:
        stem = audio_file.stem
        try:
            import librosa
            waveform, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"[efficientnet] skip {audio_file.name}: {e}", file=sys.stderr)
            continue

        clip_samples = SAMPLE_RATE * CLIP_SECS
        n_clips = max(1, len(waveform) // clip_samples)
        for clip_idx in range(n_clips):
            start = clip_idx * clip_samples
            clip_wav = waveform[start: start + clip_samples]
            if len(clip_wav) < clip_samples:
                clip_wav = np.pad(clip_wav, (0, clip_samples - len(clip_wav)))

            end_sec = (clip_idx + 1) * CLIP_SECS
            row_id  = f"{stem}_{end_sec}"

            try:
                # Build mel spectrogram for this clip
                mel = librosa.feature.melspectrogram(
                    y=clip_wav, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=16000,
                    n_fft=1024, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
                img = (mel_db * 255).astype(np.uint8)
                from torchvision import transforms
                t = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)
                preprocess = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ])
                tensor = preprocess(t / 255.0).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits    = model(tensor)
                    raw_probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

                # Map model classes → 234 BirdCLEF species
                species_probs = uniform_probs(np) * 0.01
                for model_idx, sp_idx in label_to_species_idx.items():
                    if model_idx < len(raw_probs):
                        species_probs[sp_idx] = float(raw_probs[model_idx])
                total = species_probs.sum()
                if total > 0:
                    species_probs /= total
            except Exception as e:
                print(f"[efficientnet] clip {clip_idx} error: {e}", file=sys.stderr)
                species_probs = uniform_probs(np)

            max_probs.append(float(species_probs.max()))
            rows.append((row_id, species_probs))

    score = float(np.mean(max_probs)) if max_probs else 0.0
    write_submission(args.output, rows, score)
    print(json.dumps({"score": score, "rows": len(rows)}))


def _infer_num_classes(state_dict: dict) -> int:
    """Try to infer output class count from the last classifier layer."""
    for key in reversed(list(state_dict.keys())):
        if "classifier" in key and "weight" in key:
            return state_dict[key].shape[0]
    return 182  # BirdCLEF-2026 default


def _fatal(msg: str):
    print(json.dumps({"score": 0.0, "error": msg}))
    sys.exit(1)


if __name__ == "__main__":
    main()
