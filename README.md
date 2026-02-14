# eo-sample-namer 🎵

Intelligent audio sample renamer. Analyzes audio files and generates descriptive filenames based on content.

**Before:**
```
Recording 47.wav
bounce-2-final-FINAL.wav
Untitled 3.aif
```

**After:**
```
kick_dark_punchy_short.wav
pad_warm_slow_long_Am.wav
hihat_bright_short.wav
```

## Features

- **Type classification**: kick, snare, hihat, clap, bass, pad, lead, fx, loop, pluck, stab, perc, noise
- **Character description**: warm, bright, dark, harsh, soft, punchy, wide, thin, loud, slow
- **Duration category**: short, medium, long, loop
- **Key detection**: for tonal content (C, Db, D, ... B)
- **BPM detection**: for loops
- **Batch processing**: rename entire sample libraries at once
- **Dry-run mode**: preview all renames before applying

## How It Works

Uses spectral analysis (MFCC, spectral centroid, bandwidth, rolloff), harmonic-percussive source separation (HPSS), onset detection, and envelope analysis to classify audio content. No machine learning models needed — pure DSP.

## Install

```bash
# Clone
git clone https://github.com/Cevah3ro/eo-sample-namer.git
cd eo-sample-namer

# Install dependencies (Python 3.10+)
pip install librosa click rich numpy soundfile
```

## Usage

```bash
# Analyze a single file (show features)
python eo_sample_namer.py analyze my_sample.wav

# Analyze with JSON output
python eo_sample_namer.py analyze my_sample.wav --json-output

# Rename a single file
python eo_sample_namer.py rename my_sample.wav

# Preview batch rename (dry run)
python eo_sample_namer.py batch ./samples/ --dry-run

# Batch rename all audio files
python eo_sample_namer.py batch ./samples/

# Recursive batch rename
python eo_sample_namer.py batch ./samples/ --recursive
```

## Supported Formats

WAV, MP3, AIF/AIFF, FLAC, OGG, M4A

## Classification Details

### Type Detection

| Type | Key Features |
|------|-------------|
| kick | Low centroid (<1500 Hz), fast attack, short |
| snare | Mid centroid, moderate noise (ZCR), some harmonic content |
| hihat | High ZCR (>0.2), low harmonic ratio, short |
| clap | Fully percussive (harmonic ratio ≈ 0), mid-high centroid |
| bass | Low centroid (<500 Hz), sustained |
| pad | Slow attack (>0.2s), high harmonic content |
| lead | Fast attack, high centroid (>1500 Hz), sustained |
| fx | High ZCR, slow attack, noisy texture |
| loop | Many onsets (>6), high percussive ratio, long duration |

### Character Tags

- **Brightness**: bright (>4kHz centroid) / warm (<2kHz) / dark (<1kHz)
- **Width**: wide (>3kHz bandwidth) / thin (<1kHz)
- **Energy**: loud (>0.15 RMS) / soft (<0.02 RMS)
- **Attack**: punchy (<5ms) / slow (>100ms)

## Use in Your Projects

The analysis functions can be imported as a library:

```python
from eo_sample_namer import analyze_audio, generate_name

analysis = analyze_audio("my_sample.wav")
print(analysis["type"])       # "kick"
print(analysis["character"])  # ["dark", "punchy"]
print(analysis["key"])        # "C"
print(analysis["bpm"])        # 120

new_name = generate_name(analysis, "my_sample.wav")
print(new_name)  # "kick_dark_punchy_short_C.wav"
```

## License

MIT — use it, fork it, put it in your plugins.

## Credits

Built by [Eo](https://moltbook.com/u/Eo-the-wise) 🌟 for [Cevah3ro](https://github.com/Cevah3ro)
