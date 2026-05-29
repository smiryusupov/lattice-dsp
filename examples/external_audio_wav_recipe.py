"""Recipe: bring an external WAV signal into lattice-dsp without extra dependencies.

This example uses only the Python standard library for WAV I/O plus NumPy and
``lattice-dsp``.  It is meant as a minimal bridge for speech or audio generated
outside the package, for example with eSpeak/eSpeak NG:

    espeak-ng -w speech.wav "This is a lattice DSP test signal."

For production audio I/O, use your preferred optional package such as soundfile,
librosa, or scipy.io.wavfile in your own application code.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def write_mono_pcm16(path: Path, sample_rate: int, x: np.ndarray) -> None:
    """Write a mono float signal in [-1, 1] as 16-bit PCM WAV."""

    x16 = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    payload = b"".join(struct.pack("<h", int(round(v * 32767.0))) for v in x16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(payload)


def read_mono_pcm16(path: Path) -> tuple[int, np.ndarray]:
    """Read a mono or stereo 16-bit PCM WAV as a floating NumPy array."""

    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    x = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return sample_rate, x


def synthetic_speech_like_signal(sample_rate: int, seconds: float) -> np.ndarray:
    """Generate a small deterministic voiced-speech-like test signal."""

    n = int(round(sample_rate * seconds))
    t = np.arange(n) / sample_rate
    envelope = 0.5 * (1.0 + np.sin(2.0 * math.pi * 3.0 * t))
    carrier = (
        0.55 * np.sin(2.0 * math.pi * 180.0 * t)
        + 0.25 * np.sin(2.0 * math.pi * 360.0 * t)
        + 0.12 * np.sin(2.0 * math.pi * 720.0 * t)
    )
    return 0.55 * envelope * carrier


def main() -> None:
    sample_rate = 16_000
    path = Path("external_audio_recipe_input.wav")

    # Replace this block with an externally generated file, for example:
    #   espeak-ng -w speech.wav "This is a lattice DSP test signal."
    #   path = Path("speech.wav")
    write_mono_pcm16(path, sample_rate, synthetic_speech_like_signal(sample_rate, 0.25))

    loaded_rate, x = read_mono_pcm16(path)
    filt = ld.LatticeIIR([0.35, -0.2], [0.45, 0.05, 0.2])
    y = np.asarray(filt.process(x), dtype=float)

    print("loaded sample rate:", loaded_rate)
    print("loaded samples:", x.shape[0])
    print("input RMS:", f"{np.sqrt(np.mean(x**2)):.6f}")
    print("filtered RMS:", f"{np.sqrt(np.mean(y**2)):.6f}")
    print("example WAV path:", path)


if __name__ == "__main__":
    main()
