"""Apply mdx_extra source separation three times to isolate the instrumental track.

Each pass drops the vocals estimated by Demucs' mdx_extra model; three passes
are used because a single pass still leaves audible residuals on noisy or
speech-prompted generations. Input and intermediate directories are created
next to the source with suffixes ``_mdx1`` / ``_mdx2`` / ``_3xmdx``.

Usage:
    python scripts/postproc/apply_3xmdx.py <input_dir>
"""

import sys
from pathlib import Path

import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from tqdm import tqdm


def main() -> None:
    src_base = Path(sys.argv[1])
    if not src_base.exists():
        raise FileNotFoundError(src_base)

    print(f"[3xmdx] loading mdx_extra model...", flush=True)
    mdx = get_model("mdx_extra").eval().cuda()
    src = src_base
    for p in range(3):
        suffix = f"_mdx{p + 1}" if p < 2 else "_3xmdx"
        dst = src_base.parent / (src_base.name + suffix)
        dst.mkdir(parents=True, exist_ok=True)
        wavs = sorted(src.glob("*.wav"))
        for wav_path in tqdm(wavs, desc=f"3xmdx pass {p + 1}/3", unit="wav"):
            out = dst / wav_path.name
            if out.exists():
                continue
            wav, sr = torchaudio.load(wav_path)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            with torch.no_grad():
                sources = apply_model(mdx, wav.unsqueeze(0).cuda())
            inst = sources[0, :3].sum(0).mean(0, keepdim=True).cpu()
            inst = inst / (inst.abs().max() + 1e-8) * 0.95
            torchaudio.save(str(out), inst, sr)
            torch.cuda.empty_cache()
        print(f"[3xmdx] pass {p + 1}/3 done -> {dst}", flush=True)
        src = dst


if __name__ == "__main__":
    main()
