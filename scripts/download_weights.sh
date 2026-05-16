#!/bin/bash
# Fetch every external artefact required by `run.py infer` / `run.py reproduce`.
#
# 1. Baseline weights (VAE, empty-string embeddings, vocoder) from HuggingFace
#    AndreasXi/MeanAudio  -> ./weights/
# 2. The 44 kHz VAE (v1-44.pth) from the MMAudio GitHub release  -> ./weights/
# 3. The submission checkpoint (CRPO-A, ~460 MB) from this repository's
#    GitHub Releases page  ->
#    ./exps/CRPO_A_FTexpert_CLAP/CRPO_A_FTexpert_CLAP_last.pth
#
# Usage (from the project root, inside the `meanaudio` env):
#   bash scripts/download_weights.sh
#   FORCE=1 bash scripts/download_weights.sh   # re-download even if present
set -euo pipefail

cd "$(dirname "$0")/.."

# v1-44.pth is the upstream MMAudio 44 kHz VAE — same file the MeanAudio
# baseline uses, so we pull it directly from MMAudio's GitHub release.
V1_44_URL="${V1_44_URL:-https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth}"
V1_44_MD5="${V1_44_MD5:-fab020275fa44c6589820ce025191600}"
HF_REPO="${HF_REPO:-AndreasXi/MeanAudio}"

# Submission checkpoint: published as a GitHub Release artefact on this repo.
# Override CKPT_URL to point at a specific release tag if you want a non-latest version.
CKPT_URL="${CKPT_URL:-https://github.com/yonghyunk1m/ttm-humanpref/releases/latest/download/sub1_seed42_fluxaudio_s.pt}"

FORCE="${FORCE:-0}"

echo "==> Step 1/3: text/audio encoder files from HuggingFace ($HF_REPO)"
HF_FILES=(
    empty_string_t5.pth
    empty_string_clap_c.pth
    music_speech_audioset_epoch_15_esc_89.98.pt
)
NEED_HF=0
for f in "${HF_FILES[@]}"; do
    [ -f "weights/$f" ] || NEED_HF=1
done

if [ "$NEED_HF" = "1" ] || [ "$FORCE" = "1" ]; then
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "[download] huggingface-cli not found — install huggingface_hub first:"
        echo "           pip install huggingface_hub"
        exit 1
    fi
    mkdir -p weights
    huggingface-cli download "$HF_REPO" "${HF_FILES[@]}" --local-dir weights
else
    echo "[download] encoder files already present — skipping HF download."
fi

echo
echo "==> Step 2/3: 44 kHz VAE (v1-44.pth, ~1.2 GB) from MMAudio GitHub release"
mkdir -p weights
if [ -f weights/v1-44.pth ] && [ "$FORCE" != "1" ] \
    && [ "$(md5sum weights/v1-44.pth | awk '{print $1}')" = "$V1_44_MD5" ]; then
    echo "[download] weights/v1-44.pth already present (md5 OK) — skipping."
else
    curl -L --fail --progress-bar "$V1_44_URL" -o weights/v1-44.pth \
        || { echo "[download] failed to fetch v1-44.pth from $V1_44_URL"; exit 1; }
    got="$(md5sum weights/v1-44.pth | awk '{print $1}')"
    if [ "$got" != "$V1_44_MD5" ]; then
        echo "[download] MD5 mismatch for v1-44.pth (got $got, expected $V1_44_MD5)"; exit 1
    fi
fi

echo
echo "==> Step 3/3: submission checkpoint (~460 MB) from this repo's GitHub Releases"
CKPT="exps/CRPO_A_FTexpert_CLAP/CRPO_A_FTexpert_CLAP_last.pth"
mkdir -p "$(dirname "$CKPT")"
if [ -f "$CKPT" ] && [ "$FORCE" != "1" ]; then
    echo "[download] $CKPT already present — skipping."
else
    curl -L --fail --progress-bar "$CKPT_URL" -o "$CKPT" \
        || { echo "[download] failed to fetch checkpoint from $CKPT_URL"; exit 1; }
fi

echo
echo "[download] done. Expected files:"
ls -la weights/v1-44.pth weights/empty_string_t5.pth weights/empty_string_clap_c.pth "$CKPT" 2>/dev/null || true
