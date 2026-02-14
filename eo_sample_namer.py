#!/usr/bin/env python3
"""
eo-sample-namer — Intelligent audio sample renamer.

Analyzes audio files and generates descriptive filenames based on content:
- Type classification (kick, snare, hihat, clap, pad, lead, bass, fx, etc.)
- Character description (warm, bright, dark, harsh, soft, punchy, etc.)
- Duration category (short, medium, long)
- Musical key detection (for tonal content)
- BPM detection (for loops)

Usage:
    python eo_sample_namer.py analyze <file>       # Show analysis
    python eo_sample_namer.py rename <file>        # Rename single file
    python eo_sample_namer.py batch <directory>    # Rename all files in dir
    python eo_sample_namer.py batch <dir> --dry-run  # Preview renames

MIT License — github.com/Cevah3ro/eo-sample-namer
"""

import os
import sys
import json
import shutil
import numpy as np
import click
from rich.console import Console
from rich.table import Table

console = Console()

# Lazy imports for speed
_librosa = None

def get_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


# ─── Audio Analysis ───────────────────────────────────────────────

def load_audio(filepath, sr=22050, duration=None):
    """Load audio file, return (y, sr)."""
    librosa = get_librosa()
    y, sr = librosa.load(filepath, sr=sr, duration=duration)
    return y, sr


def analyze_audio(filepath):
    """Full analysis of an audio file. Returns dict of features."""
    librosa = get_librosa()
    y, sr = load_audio(filepath)

    if len(y) == 0:
        return {"error": "Empty audio file"}

    analysis = {}

    # Duration
    duration = len(y) / sr
    analysis["duration_sec"] = round(duration, 3)
    analysis["duration_cat"] = classify_duration(duration)

    # RMS energy
    rms = np.sqrt(np.mean(y**2))
    analysis["rms"] = round(float(rms), 4)

    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y))

    analysis["spectral_centroid"] = round(float(spectral_centroid), 1)
    analysis["spectral_rolloff"] = round(float(spectral_rolloff), 1)
    analysis["spectral_bandwidth"] = round(float(spectral_bandwidth), 1)
    analysis["zero_crossing_rate"] = round(float(zero_crossings), 4)

    # Onset strength (transient detection)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    analysis["onset_strength_mean"] = round(float(np.mean(onset_env)), 2)
    analysis["onset_strength_max"] = round(float(np.max(onset_env)), 2)
    analysis["onset_count"] = int(len(librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)))

    # Envelope shape (attack/decay)
    envelope = np.abs(y)
    # Smooth envelope with smaller kernel for transient detection
    kernel_size = min(256, len(envelope))
    if kernel_size > 0:
        envelope_smooth = np.convolve(envelope, np.ones(kernel_size)/kernel_size, mode='same')
        peak_idx = np.argmax(envelope_smooth)
        attack_time = peak_idx / sr
        analysis["attack_time"] = round(attack_time, 4)
        # More generous percussive detection
        analysis["is_percussive"] = attack_time < 0.05 and duration < 2.0
    else:
        analysis["attack_time"] = 0
        analysis["is_percussive"] = False

    # Noise ratio (high ZCR + low periodicity = noisy)
    analysis["noise_ratio"] = round(float(zcr_val := zero_crossings), 4)

    # Harmonic-percussive separation
    S = np.abs(librosa.stft(y))
    H, P = librosa.decompose.hpss(S)
    harmonic_energy = np.sum(H**2)
    percussive_energy = np.sum(P**2)
    total = harmonic_energy + percussive_energy + 1e-10
    analysis["harmonic_ratio"] = round(float(harmonic_energy / total), 3)
    analysis["percussive_ratio"] = round(float(percussive_energy / total), 3)

    # Key detection (chroma)
    if duration > 0.5:  # Only for longer samples
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_mean)
        key_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        analysis["key"] = key_names[key_idx]
        analysis["key_confidence"] = round(float(chroma_mean[key_idx] / np.sum(chroma_mean)), 3)
    else:
        analysis["key"] = None
        analysis["key_confidence"] = 0

    # BPM (for loops)
    if duration > 1.0:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        analysis["bpm"] = round(bpm_val)
    else:
        analysis["bpm"] = None

    # MFCCs for timbre classification
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    analysis["mfcc_mean"] = [round(float(x), 2) for x in np.mean(mfccs, axis=1)]

    # Classification
    analysis["type"] = classify_type(analysis)
    analysis["character"] = classify_character(analysis)

    return analysis


# ─── Classification ───────────────────────────────────────────────

def classify_duration(duration):
    """Classify duration into categories."""
    if duration < 0.3:
        return "short"
    elif duration < 2.0:
        return "medium"
    elif duration < 8.0:
        return "long"
    else:
        return "loop"


def classify_type(a):
    """Classify sample type based on audio features."""
    dur = a["duration_sec"]
    centroid = a["spectral_centroid"]
    attack = a["attack_time"]
    is_perc = a["is_percussive"]
    zcr = a["zero_crossing_rate"]
    rms = a["rms"]
    onset_count = a["onset_count"]
    perc_ratio = a.get("percussive_ratio", 0)
    harm_ratio = a.get("harmonic_ratio", 0)

    # ─── FX detection first (noisy + slow attack = fx/riser/texture) ───
    if zcr > 0.15 and attack > 0.1 and dur > 0.5:
        return "fx"

    # ─── Short percussive sounds (< 1s, fast attack) ───
    is_short_perc = (is_perc or (perc_ratio > 0.4 and dur < 1.0)) and attack < 0.05 and dur < 1.0

    if is_short_perc:
        # ─── HIHAT: very high centroid + very low harmonic content ───
        # Hihats are almost pure noise (harm<0.15) with very high centroid
        # OR extremely noisy (zcr>0.5) with high centroid
        if centroid > 5000 and harm_ratio < 0.15:
            return "hihat"
        if zcr > 0.5 and centroid > 3000 and harm_ratio < 0.3:
            return "hihat"

        # ─── KICK: low centroid, sub-heavy ───
        if centroid < 500 and harm_ratio > 0.2:
            return "kick"
        if centroid < 300:
            return "kick"

        # ─── CLAP: mid-high centroid, very noisy (harm<0.1), bandpassed feel ───
        if harm_ratio < 0.1 and 2000 < centroid < 6000 and zcr > 0.15:
            return "clap"

        # ─── SNARE: noise+tone mix, mid-high centroid ───
        if 0.1 < harm_ratio < 0.6 and centroid > 1000 and zcr > 0.05:
            return "snare"

        # ─── KICK: broader low centroid catch ───
        if centroid < 1500 and harm_ratio > 0.3:
            return "kick"

        # ─── Tonal percussion (congas, toms, etc): harmonic, mid centroid ───
        if harm_ratio > 0.3 and centroid > 150:
            return "perc"

        # ─── Remaining noisy short sounds ───
        if centroid > 3000 and harm_ratio < 0.3:
            return "hihat"

        return "perc"

    # ─── Loops (many onsets + percussive + long) ───
    if dur > 2.0 and onset_count >= 6 and perc_ratio > 0.5:
        return "loop"

    # ─── Tonal / sustained sounds (>= 1.5s) ───
    if dur >= 1.5:
        if centroid < 500 and attack < 0.1:
            return "bass"
        elif attack > 0.2:
            # Very slow attack = pad regardless of centroid
            return "pad"
        elif harm_ratio > 0.8 and attack > 0.07 and onset_count <= 3:
            return "pad"
        elif centroid > 1500:
            return "lead"
        elif centroid < 500 and attack >= 0.1:
            return "pad"
        else:
            return "lead"

    # ─── Medium-length sounds (0.3 - 1.5s) ───
    if 0.3 <= dur < 1.5:
        if centroid < 500:
            return "bass"
        elif zcr > 0.15:
            return "fx"
        elif attack < 0.01:
            return "pluck"
        elif centroid > 3000:
            return "lead"
        else:
            return "stab"

    # ─── Short non-percussive ───
    if zcr > 0.3:
        return "noise"
    return "oneshot"


def classify_character(a):
    """Classify tonal character/texture."""
    centroid = a["spectral_centroid"]
    bandwidth = a["spectral_bandwidth"]
    rolloff = a["spectral_rolloff"]
    rms = a["rms"]

    chars = []

    # Brightness
    if centroid > 4000:
        chars.append("bright")
    elif centroid < 1000:
        chars.append("dark")
    elif centroid < 2000:
        chars.append("warm")

    # Width
    if bandwidth > 3000:
        chars.append("wide")
    elif bandwidth < 1000:
        chars.append("thin")

    # Energy
    if rms > 0.15:
        chars.append("loud")
    elif rms < 0.02:
        chars.append("soft")

    # Attack character
    if a["attack_time"] < 0.005:
        chars.append("punchy")
    elif a["attack_time"] > 0.1:
        chars.append("slow")

    if not chars:
        chars.append("neutral")

    return chars


# ─── Naming ───────────────────────────────────────────────────────

def generate_name(analysis, original_path):
    """Generate a descriptive filename from analysis."""
    ext = os.path.splitext(original_path)[1].lower()

    parts = []

    # Type
    parts.append(analysis["type"])

    # Character (max 2)
    chars = analysis["character"][:2]
    parts.extend(chars)

    # Duration category
    parts.append(analysis["duration_cat"])

    # Key (if detected with confidence)
    if analysis.get("key") and analysis.get("key_confidence", 0) > 0.1:
        parts.append(analysis["key"])

    # BPM (if detected and it's a loop)
    if analysis.get("bpm") and analysis["duration_cat"] == "loop":
        parts.append(f"{analysis['bpm']}bpm")

    name = "_".join(parts) + ext
    return name


# ─── CLI ──────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aif', '.aiff', '.flac', '.ogg', '.m4a'}


@click.group()
@click.version_option(version="0.1.0", prog_name="eo-sample-namer")
def cli():
    """🎵 eo-sample-namer — Intelligent audio sample renamer."""
    pass


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def analyze(filepath, json_output):
    """Analyze a single audio file and show features."""
    analysis = analyze_audio(filepath)

    if json_output:
        click.echo(json.dumps(analysis, indent=2))
        return

    table = Table(title=f"🎵 Analysis: {os.path.basename(filepath)}")
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green")

    suggested = generate_name(analysis, filepath)
    table.add_row("Suggested Name", f"[bold]{suggested}[/bold]")
    table.add_row("─" * 20, "─" * 30)

    for key, val in analysis.items():
        if key == "mfcc_mean":
            continue  # Skip raw MFCCs in display
        table.add_row(key, str(val))

    console.print(table)


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--dry-run", "-n", is_flag=True, help="Preview without renaming")
def rename(filepath, dry_run):
    """Rename a single audio file based on analysis."""
    analysis = analyze_audio(filepath)
    new_name = generate_name(analysis, filepath)
    directory = os.path.dirname(filepath) or "."
    new_path = os.path.join(directory, new_name)

    # Avoid collisions
    new_path = resolve_collision(new_path)

    if dry_run:
        console.print(f"[yellow]Would rename:[/yellow] {os.path.basename(filepath)} → {os.path.basename(new_path)}")
    else:
        os.rename(filepath, new_path)
        console.print(f"[green]Renamed:[/green] {os.path.basename(filepath)} → {os.path.basename(new_path)}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--dry-run", "-n", is_flag=True, help="Preview without renaming")
@click.option("--recursive", "-r", is_flag=True, help="Process subdirectories")
def batch(directory, dry_run, recursive):
    """Rename all audio files in a directory."""
    files = []
    if recursive:
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(directory, f))

    if not files:
        console.print("[yellow]No audio files found.[/yellow]")
        return

    console.print(f"[cyan]Found {len(files)} audio files.[/cyan]")

    table = Table(title="🎵 Batch Rename Preview" if dry_run else "🎵 Batch Rename")
    table.add_column("Original", style="dim")
    table.add_column("→", style="yellow")
    table.add_column("New Name", style="green")

    for filepath in sorted(files):
        try:
            analysis = analyze_audio(filepath)
            new_name = generate_name(analysis, filepath)
            rel_dir = os.path.dirname(filepath)
            new_path = resolve_collision(os.path.join(rel_dir, new_name))

            table.add_row(
                os.path.basename(filepath),
                "→",
                os.path.basename(new_path)
            )

            if not dry_run:
                os.rename(filepath, new_path)

        except Exception as e:
            table.add_row(
                os.path.basename(filepath),
                "⚠",
                f"[red]Error: {e}[/red]"
            )

    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run — no files were renamed. Remove --dry-run to apply.[/yellow]")
    else:
        console.print(f"\n[green]✅ Renamed {len(files)} files.[/green]")


def resolve_collision(path):
    """Add numeric suffix if file already exists."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 2
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"


if __name__ == "__main__":
    cli()
