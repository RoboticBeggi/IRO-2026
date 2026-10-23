"""Speech-to-text speed check on the Jetson: Parakeet-TDT 0.6B v2 (sherpa-onnx) on the voice samples.
    ~/.venvs/voice/bin/python tools/stt_bench.py
Real accuracy needs real voices (recorded through the mic array); this checks speed and sanity."""

import glob
import time
from pathlib import Path

import numpy as np
import soundfile as sf

M = Path.home() / "voice-models" / "parakeet-v2"


def resample(a, sr, to=16000):
    if sr == to:
        return a
    n = int(len(a) * to / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def main():
    import sherpa_onnx
    t0 = time.time()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(M / "encoder.int8.onnx"), decoder=str(M / "decoder.int8.onnx"),
        joiner=str(M / "joiner.int8.onnx"), tokens=str(M / "tokens.txt"),
        num_threads=4, model_type="nemo_transducer", provider="cpu")
    print(f"parakeet loaded in {time.time() - t0:.1f}s")
    for f in sorted(glob.glob(str(Path.home() / "voice-samples/kokoro-af_heart/line*.wav"))):
        a, sr = sf.read(f, dtype="float32")
        a = resample(a, sr)
        s = rec.create_stream()
        s.accept_waveform(16000, a)
        t = time.time()
        rec.decode_stream(s)
        dt = time.time() - t
        print(f"{Path(f).name}: {dt:.2f}s for {len(a) / 16000:.1f}s audio (RTF {dt / (len(a) / 16000):.2f}) -> {s.result.text}")


if __name__ == "__main__":
    main()
