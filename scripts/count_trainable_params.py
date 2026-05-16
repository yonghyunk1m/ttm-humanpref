"""
Count the number of trainable parameters of the submitted model
(ICME 2026 ATTM Efficiency Track).

Usage:
    python scripts/count_trainable_params.py    # from the repo root
"""
import os
import sys

# Make the repo importable even when the installed `meanaudio` egg-link still
# points at a different checkout of this project.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from meanaudio.model.networks_score import SCORE_MODEL_REGISTRY

# Submission backbone: fluxaudio_score_s_v2_44k
# (CRPO-A fine-tune of FT_expert_only_r1; see scripts/flowmatching/train_crpo_A_from_FT_expert.sh)
MODEL_NAME = "fluxaudio_score_s_v2_44k"
TEXT_C_DIM = 512  # CLAP pooled (see config/base_config.yaml)
USE_ROPE = True   # see scripts/flowmatching/train_crpo_A_from_FT_expert.sh


def human(n: int) -> str:
    return f"{n:,}  ({n / 1e6:.2f} M)"


def main() -> None:
    net = SCORE_MODEL_REGISTRY[MODEL_NAME](
        text_c_dim=TEXT_C_DIM,
        use_rope=USE_ROPE,
    )

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Track          : ICME 2026 ATTM Efficiency")
    print(f"Model          : {MODEL_NAME}  (text_c_dim={TEXT_C_DIM}, use_rope={USE_ROPE})")
    print(f"--------------------------------------------------")
    print(f"Total params   : {human(total)}")
    print(f"Trainable      : {human(trainable)}")
    print(f"Frozen (buffers): {human(frozen)}")


if __name__ == "__main__":
    main()
