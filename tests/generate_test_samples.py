#!/usr/bin/env python3
"""Generate synthetic test audio samples for eo-sample-namer testing."""

import numpy as np
import os
import sys

# Add soundfile
sys.path.insert(0, "/home/ceva/audio-env/lib/python3.12/site-packages")
import soundfile as sf

OUTDIR = os.path.join(os.path.dirname(__file__), "samples")
SR = 44100

os.makedirs(OUTDIR, exist_ok=True)


def save(name, y, sr=SR):
    path = os.path.join(OUTDIR, name)
    sf.write(path, y, sr)
    print(f"  Created: {name} ({len(y)/sr:.2f}s)")


def generate_kick():
    """Low-frequency transient — classic kick drum."""
    t = np.linspace(0, 0.3, int(SR * 0.3))
    freq = 150 * np.exp(-t * 15)  # Pitch drop
    envelope = np.exp(-t * 12)
    y = np.sin(2 * np.pi * freq * t) * envelope * 0.9
    save("test_kick.wav", y)


def generate_snare():
    """Mid-frequency transient + noise — snare drum."""
    t = np.linspace(0, 0.25, int(SR * 0.25))
    # Tone component
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 20) * 0.5
    # Noise component
    noise = np.random.randn(len(t)) * np.exp(-t * 15) * 0.4
    y = tone + noise
    save("test_snare.wav", y)


def generate_hihat():
    """High-frequency noise burst — hihat."""
    t = np.linspace(0, 0.1, int(SR * 0.1))
    noise = np.random.randn(len(t)) * np.exp(-t * 40) * 0.3
    # Highpass-ish: differentiate
    y = np.diff(noise, prepend=0) * 5
    save("test_hihat.wav", y)


def generate_clap():
    """Multiple short noise bursts — clap."""
    t = np.linspace(0, 0.2, int(SR * 0.2))
    y = np.zeros_like(t)
    for offset in [0.0, 0.01, 0.02, 0.025]:
        idx = int(offset * SR)
        burst_len = int(0.03 * SR)
        if idx + burst_len <= len(t):
            burst = np.random.randn(burst_len) * np.exp(-np.linspace(0, 3, burst_len))
            y[idx:idx+burst_len] += burst * 0.3
    save("test_clap.wav", y)


def generate_bass():
    """Low sustained tone — bass."""
    t = np.linspace(0, 1.5, int(SR * 1.5))
    envelope = np.minimum(t * 50, 1.0) * np.exp(-t * 1.5)
    y = np.sin(2 * np.pi * 60 * t) * envelope * 0.7
    # Add harmonics
    y += np.sin(2 * np.pi * 120 * t) * envelope * 0.3
    save("test_bass.wav", y)


def generate_pad():
    """Slow attack, sustained, warm — pad sound."""
    t = np.linspace(0, 4.0, int(SR * 4.0))
    # Slow attack
    attack = np.minimum(t / 0.5, 1.0)
    release = np.maximum(1.0 - (t - 3.0) / 1.0, 0.0)
    envelope = np.minimum(attack, release)
    # Rich tone (multiple harmonics)
    y = np.sin(2 * np.pi * 220 * t) * 0.4
    y += np.sin(2 * np.pi * 330 * t) * 0.25
    y += np.sin(2 * np.pi * 440 * t) * 0.15
    y += np.sin(2 * np.pi * 550 * t) * 0.1
    y *= envelope * 0.6
    save("test_pad.wav", y)


def generate_lead():
    """Bright sustained tone — lead synth."""
    t = np.linspace(0, 2.0, int(SR * 2.0))
    envelope = np.minimum(t * 20, 1.0) * np.exp(-t * 0.5)
    # Sawtooth-ish (bright)
    y = np.zeros_like(t)
    for h in range(1, 10):
        y += np.sin(2 * np.pi * 880 * h * t) / h
    y *= envelope * 0.3
    save("test_lead.wav", y)


def generate_fx():
    """Noisy, evolving texture — FX/riser."""
    t = np.linspace(0, 1.5, int(SR * 1.5))
    # Rising noise
    freq = np.linspace(200, 8000, len(t))
    y = np.sin(2 * np.pi * np.cumsum(freq / SR)) * 0.4
    y += np.random.randn(len(t)) * 0.15
    envelope = t / 1.5  # Rising
    y *= envelope
    save("test_fx.wav", y)


def generate_loop():
    """Simple 4-beat loop at 120 BPM."""
    bpm = 120
    beat_len = int(SR * 60 / bpm)
    n_beats = 8
    y = np.zeros(beat_len * n_beats)
    t_kick = np.linspace(0, 0.15, int(SR * 0.15))
    kick = np.sin(2 * np.pi * 100 * np.exp(-t_kick * 10) * t_kick) * np.exp(-t_kick * 15) * 0.7

    for i in range(n_beats):
        idx = i * beat_len
        end = min(idx + len(kick), len(y))
        y[idx:end] += kick[:end-idx]
        # Hihat on every beat
        hh_len = min(int(SR * 0.05), len(y) - idx)
        y[idx:idx+hh_len] += np.random.randn(hh_len) * np.exp(-np.linspace(0, 10, hh_len)) * 0.15

    save("test_loop_120bpm.wav", y)


if __name__ == "__main__":
    print("🎵 Generating test samples...")
    generate_kick()
    generate_snare()
    generate_hihat()
    generate_clap()
    generate_bass()
    generate_pad()
    generate_lead()
    generate_fx()
    generate_loop()
    print(f"\n✅ Done! Samples in: {OUTDIR}")
