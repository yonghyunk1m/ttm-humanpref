#!/bin/bash
# Expert Iteration Round 1: Finetune v1_44k on combined (original + top 10% self-play) data
# The 64 expert samples have actual reward 0.85~1.48 — the model learns what high reward REALLY looks like
set -e
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."

WEIGHTS=exps/Score_v1_44k/Score_v1_44k_shadow.pth
EXP_ID=FT_v1_44k_expert_iter_r1

source "${CONDA_PREFIX%/envs/*}/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate meanaudio
python -m torch.distributed.run --standalone --nproc_per_node=1 \
    train.py \
    --config-name train_config_jamendo_44k_score.yaml \
    exp_id=$EXP_ID compile=False model=fluxaudio_score_s_v1_44k \
    batch_size=32 eval_batch_size=1 \
    num_iterations=30_000 \
    text_encoder_name=t5_clap \
    data_dim.text_c_dim=512 \
    pin_memory=False num_workers=10 ac_oversample_rate=5 \
    use_score=True use_meanflow=False null_score_prob=0.1 \
    cfg_strength=4.5 enable_grad_scaler=False \
    learning_rate=0.00001 \
    lr_schedule=constant \
    linear_warmup_steps=0 \
    ++grad_accum_steps=2 ++score_column=instrumental_reward_score \
    ++use_rope=True ++use_wandb=True ++debug=False \
    ++data.Jamendo_train_npz.tsv=data/expert_iter_r1/combined_npz.tsv \
    ++data.Jamendo_train_npz.npz_dir=data/expert_iter_r1/combined_npz \
    weights=$WEIGHTS
