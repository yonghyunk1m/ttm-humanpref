"""Normalize integrated loudness of every wav in a directory to a target LUFS.

Uses pyloudnorm; falls back to the raw waveform if loudness measurement fails
(e.g. for near-silent clips). Peaks above 0.99 are attenuated to 0.95 to avoid
clipping after normalisation.

Usage:
    python scripts/postproc/apply_lufs.py <src_dir> <out_dir> <target_lufs>

Example (submission pipeline):
    python scripts/postproc/apply_lufs.py out/_3xmdx out/_lufs165 -16.5
"""

import os
import sys
import warnings

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

# pyloudnorm prints "Possible clipped samples" for almost every file; our code
# attenuates peaks >0.99 to 0.95 right after, so the warning is a false alarm.
warnings.filterwarnings("ignore", message="Possible clipped samples in output.")


def main() -> None:
    src, out, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    meter = pyln.Meter(44100)
    files = sorted(f for f in os.listdir(src) if f.endswith(".wav"))
    for fn in tqdm(files, desc=f"LUFS->{target}", unit="wav"):
        w, sr = sf.read(os.path.join(src, fn))
        if w.ndim > 1:
            w = w.mean(1)
        try:
            lu = meter.integrated_loudness(w)
            wn = pyln.normalize.loudness(w, lu, target)
            peak = np.abs(wn).max()
            if peak > 0.99:
                wn = wn * 0.95 / peak
            sf.write(os.path.join(out, fn), wn.astype(np.float32), sr)
        except Exception:
            sf.write(os.path.join(out, fn), w.astype(np.float32), sr)
    print(f"[LUFS->{target}] {len(files)} -> {out}")


if __name__ == "__main__":
    main()
