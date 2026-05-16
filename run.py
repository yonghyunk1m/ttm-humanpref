"""Unified CLI for our ICME26 ATTM Grand Challenge submission.

Subcommands
-----------
    params       Print trainable / total / frozen parameter counts of the
                 submitted network (for the Efficiency Track screenshot).

    infer        Run text-to-music inference for one checkpoint on a prompt
                 CSV. Thin wrapper over ``scripts/inference_crossover.py``.

    postproc     Apply the 3xmdx vocal-removal and LUFS-normalise passes to a
                 directory of generated wavs.

    eval         Evaluate a directory of wavs against a GT set (FAD + CLAP),
                 via ``scripts/eval_crossover_ab.py``.

    reproduce    End-to-end: generate 100 wavs for the given seed (42 or 55),
                 apply 3xmdx + LUFS-16.5, and zip them as the submission file.

    train        Launch a training stage:
                   ``pretrain``        score-conditioned FluxAudio-S 44k
                   ``expert-iter-r1``  FT_expert_only_r1 fine-tune
                   ``crpo-a``          CRPO-A reward-preference fine-tune
                 (Shells out to ``scripts/flowmatching/*.sh``.)

Examples
--------
    # 0. Efficiency-Track parameter screenshot
    python run.py params

    # 1. Re-create Submission 1 (CRPO-A seed 42)
    python run.py reproduce --seed 42 --output output/efficiency_sub1.zip

    # 2. Re-create Submission 2 (CRPO-A seed 55)
    python run.py reproduce --seed 55 --output output/efficiency_sub2.zip

    # 3. Single-prompt inference with the submission model
    python run.py infer \
        --weights exps/CRPO_A_FTexpert_CLAP/CRPO_A_FTexpert_CLAP_last.pth \
        --prompts_csv data/test/final_test_prompts.csv \
        --output out/custom --seed 42 --prefix "high quality instrumental music, "

    # 4. Train stage 3 (CRPO-A) from the expert-iter-r1 checkpoint
    CUDA_VISIBLE_DEVICES=5 python run.py train --stage crpo-a
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

SUBMISSION_MODEL = "fluxaudio_score_s_v2_44k"
SUBMISSION_WEIGHTS = "exps/CRPO_A_FTexpert_CLAP/CRPO_A_FTexpert_CLAP_last.pth"
SUBMISSION_PREFIX = "high quality instrumental music, "
SUBMISSION_SCORE = 5.0
SUBMISSION_CFG = 4.0
SUBMISSION_NUM_STEPS = 25
SUBMISSION_LUFS = -16.5
SUBMISSION_PROMPTS = "data/test/final_test_prompts.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _run(cmd: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    # Ensure child processes find the repo's local `meanaudio` package even if
    # the env's egg-link still points at a different checkout.
    env = kwargs.pop("env", os.environ.copy())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    try:
        subprocess.check_call(cmd, cwd=REPO_ROOT, env=env, **kwargs)
    except subprocess.CalledProcessError:
        _maybe_hint_missing_env()
        raise


def _maybe_hint_missing_env() -> None:
    """Print a conda-env hint when torch is not importable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "\n[run.py] It looks like PyTorch is not installed in the current Python "
            "environment. Set up the project env first:\n"
            "    bash setup.sh                    # creates conda env 'meanaudio'\n"
            "    conda activate meanaudio\n"
            "    python run.py params\n",
            file=sys.stderr,
        )


def _zip_wavs(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for wav in sorted(src_dir.glob("*.wav")):
            z.write(wav, arcname=wav.name)
    print(f"[zip] {zip_path} ({sum(1 for _ in zipfile.ZipFile(zip_path).namelist())} files)")


def _require_inference_weights(checkpoint: str) -> None:
    """Fail fast with an actionable message if required weights are missing."""
    required = {
        "44 kHz VAE v1-44.pth (MMAudio GitHub release)":
            REPO_ROOT / "weights/v1-44.pth",
        "Empty-T5 embedding (HuggingFace AndreasXi/MeanAudio)":
            REPO_ROOT / "weights/empty_string_t5.pth",
        "Empty-CLAP embedding (HuggingFace AndreasXi/MeanAudio)":
            REPO_ROOT / "weights/empty_string_clap_c.pth",
        "LAION CLAP music_speech (HuggingFace AndreasXi/MeanAudio)":
            REPO_ROOT / "weights/music_speech_audioset_epoch_15_esc_89.98.pt",
        "Submission checkpoint (Google Drive)":
            REPO_ROOT / checkpoint,
    }
    missing = [(label, path) for label, path in required.items() if not path.exists()]
    if not missing:
        return
    lines = ["[run.py] Cannot start inference — required weight files are missing:"]
    for label, path in missing:
        lines.append(f"    - {path}   ({label})")
    lines.append("")
    lines.append("Fix with one command:")
    lines.append("    python run.py download-weights")
    lines.append("")
    lines.append("See the \"Downloading the Weights\" section of README.md for details.")
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(1)


# -----------------------------------------------------------------------------
# Subcommands
# -----------------------------------------------------------------------------


def cmd_params(_: argparse.Namespace) -> None:
    _run([sys.executable, "scripts/count_trainable_params.py"])


def cmd_download_weights(_: argparse.Namespace) -> None:
    _run(["bash", "scripts/download_weights.sh"])


def cmd_infer(args: argparse.Namespace) -> None:
    _require_inference_weights(args.weights)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/inference_crossover.py",
        "--slot1_variant", args.variant,
        "--slot1_weights", args.weights,
        "--prompts_csv", args.prompts_csv,
        "--output", str(out_dir),
        "--mode", "single",
        "--num_steps", str(args.num_steps),
        "--score1", str(args.score),
        "--cfg_mode", "constant",
        "--cfg_hi", str(args.cfg),
        "--seed_base", str(args.seed),
    ]
    if args.prefix:
        cmd += ["--prompt_prefix", args.prefix]
    _run(cmd)


def cmd_postproc(args: argparse.Namespace) -> None:
    src = Path(args.input).resolve()
    _run([sys.executable, "scripts/postproc/apply_3xmdx.py", str(src)])

    three_mdx = src.parent / (src.name + "_3xmdx")
    out = Path(args.output).resolve() if args.output else (src.parent / (src.name + "_lufs"))
    _run([sys.executable, "scripts/postproc/apply_lufs.py",
          str(three_mdx), str(out), str(args.lufs)])
    print(f"[postproc] final: {out}")


def cmd_eval(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "scripts/eval_crossover_ab.py",
        "--dirs", args.pred,
        "--labels", args.label,
        "--skip_mdx",
        "--prompts_csv", args.prompts_csv,
        "--out_json", args.out_json,
    ]
    _run(cmd)


def cmd_reproduce(args: argparse.Namespace) -> None:
    _require_inference_weights(args.weights)
    if args.seed not in (42, 55):
        print(f"[reproduce] WARNING: submission used seeds 42/55; got {args.seed}. "
              "Continuing for research use.", flush=True)

    workdir = Path(args.workdir).resolve()
    raw = workdir / f"crpoA_seed{args.seed}"
    raw.mkdir(parents=True, exist_ok=True)

    print("\n[reproduce] ===== step 1/4: generate 100 wavs (~2-3 min) =====", flush=True)
    _run([
        sys.executable, "scripts/inference_crossover.py",
        "--slot1_variant", SUBMISSION_MODEL,
        "--slot1_weights", args.weights,
        "--prompts_csv", args.prompts_csv,
        "--output", str(raw),
        "--mode", "single",
        "--num_steps", str(SUBMISSION_NUM_STEPS),
        "--score1", str(SUBMISSION_SCORE),
        "--cfg_mode", "constant",
        "--cfg_hi", str(SUBMISSION_CFG),
        "--seed_base", str(args.seed),
        "--prompt_prefix", SUBMISSION_PREFIX,
    ])

    print("\n[reproduce] ===== step 2/4: 3xmdx instrumental extraction (~4-5 min) =====", flush=True)
    _run([sys.executable, "scripts/postproc/apply_3xmdx.py", str(raw)])
    three_mdx = raw.parent / (raw.name + "_3xmdx")

    print("\n[reproduce] ===== step 3/4: LUFS normalisation =====", flush=True)
    lufs_dir = raw.parent / (raw.name + f"_lufs{str(SUBMISSION_LUFS).replace('-','m').replace('.','')}")
    _run([sys.executable, "scripts/postproc/apply_lufs.py",
          str(three_mdx), str(lufs_dir), str(SUBMISSION_LUFS)])

    print("\n[reproduce] ===== step 4/4: zipping =====", flush=True)
    zip_path = Path(args.output).resolve()
    _zip_wavs(lufs_dir, zip_path)
    print(f"\n[reproduce] SUCCESS -> {zip_path}")


def cmd_train(args: argparse.Namespace) -> None:
    stage_to_script = {
        "pretrain":       "scripts/flowmatching/train_fluxaudio_s_44k_score_v2.sh",
        "expert-iter-r1": "scripts/flowmatching/finetune_v1_44k_expert_iter.sh",
        "crpo-a":         "scripts/flowmatching/train_crpo_A_from_FT_expert.sh",
    }
    if args.stage not in stage_to_script:
        raise SystemExit(f"Unknown stage {args.stage!r}. "
                         f"Available: {sorted(stage_to_script)}")
    script = stage_to_script[args.stage]
    if not (REPO_ROOT / script).exists():
        raise SystemExit(f"Missing training script: {script}")
    _run(["bash", script])


# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="ICME 2026 ATTM Grand Challenge submission CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # params
    sp = sub.add_parser("params", help="Print trainable parameter count.")
    sp.set_defaults(func=cmd_params)

    # download-weights
    sp = sub.add_parser("download-weights",
                        help="Fetch HF baseline weights + CRPO-A submission checkpoint.")
    sp.set_defaults(func=cmd_download_weights)

    # infer
    sp = sub.add_parser("infer", help="Run text-to-music inference.")
    sp.add_argument("--variant", default=SUBMISSION_MODEL)
    sp.add_argument("--weights", default=SUBMISSION_WEIGHTS)
    sp.add_argument("--prompts_csv", default=SUBMISSION_PROMPTS)
    sp.add_argument("--output", required=True)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--score", type=float, default=SUBMISSION_SCORE)
    sp.add_argument("--cfg", type=float, default=SUBMISSION_CFG)
    sp.add_argument("--num_steps", type=int, default=SUBMISSION_NUM_STEPS)
    sp.add_argument("--prefix", default=SUBMISSION_PREFIX,
                    help='Prompt prefix. Pass "" to disable.')
    sp.set_defaults(func=cmd_infer)

    # postproc
    sp = sub.add_parser("postproc",
                        help="Apply 3xmdx + LUFS normalisation to a wav directory.")
    sp.add_argument("--input", required=True, help="Directory of raw wavs.")
    sp.add_argument("--output", default=None, help="Output directory (default: <input>_lufs).")
    sp.add_argument("--lufs", type=float, default=SUBMISSION_LUFS)
    sp.set_defaults(func=cmd_postproc)

    # eval
    sp = sub.add_parser("eval", help="Evaluate FAD + CLAP of a prediction directory.")
    sp.add_argument("--pred", required=True, help="Prediction wav directory.")
    sp.add_argument("--prompts_csv", default=SUBMISSION_PROMPTS)
    sp.add_argument("--label", default="preds")
    sp.add_argument("--out_json", default="eval_result.json")
    sp.set_defaults(func=cmd_eval)

    # reproduce
    sp = sub.add_parser("reproduce",
                        help="End-to-end reproduce one submission zip (seed 42 or 55).")
    sp.add_argument("--seed", type=int, required=True, choices=[42, 55])
    sp.add_argument("--weights", default=SUBMISSION_WEIGHTS)
    sp.add_argument("--prompts_csv", default=SUBMISSION_PROMPTS)
    sp.add_argument("--workdir", default="output/reproduce",
                    help="Scratch directory for intermediate wavs.")
    sp.add_argument("--output", required=True,
                    help="Final zip path, e.g. output/efficiency_sub1.zip.")
    sp.set_defaults(func=cmd_reproduce)

    # train
    sp = sub.add_parser("train", help="Launch a training stage.")
    sp.add_argument("--stage", required=True,
                    choices=["pretrain", "expert-iter-r1", "crpo-a"])
    sp.set_defaults(func=cmd_train)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
