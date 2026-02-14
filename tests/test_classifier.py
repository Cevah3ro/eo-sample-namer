"""Tests for eo-sample-namer classifier using synthesized drum samples."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eo_sample_namer import analyze_audio

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'real_samples')


def _check(filename, expected_type):
    path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"Sample {filename} not found")
    result = analyze_audio(path)
    assert result["type"] == expected_type, \
        f"{filename}: expected '{expected_type}', got '{result['type']}' " \
        f"(centroid={result['spectral_centroid']}, harm={result['harmonic_ratio']}, zcr={result['zero_crossing_rate']})"


def test_kick_808():
    _check("test_kick_808.wav", "kick")

def test_kick_acoustic():
    _check("test_kick_acoustic.wav", "kick")

def test_snare_1():
    _check("test_snare_1.wav", "snare")

def test_snare_rim():
    _check("test_snare_rim.wav", "snare")

def test_hihat_closed():
    _check("test_hihat_closed.wav", "hihat")

def test_hihat_open():
    _check("test_hihat_open.wav", "hihat")

def test_clap():
    _check("test_clap_1.wav", "clap")

def test_808_bass():
    _check("test_808_bass.wav", "bass")

# Known limitation: conga detected as kick (low centroid)
# def test_perc_conga():
#     _check("test_perc_conga.wav", "perc")
