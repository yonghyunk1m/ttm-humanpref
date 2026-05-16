#!/bin/bash
# CRPO Round-A (Jamendo captions, CLAP-reward) from FT_expert_only_r1.
# Expects pairs already generated at data/crpo_pairs/FT_expert_CRPO_A_CLAP.
#
# Usage: CUDA_VISIBLE_DEVICES=5 bash scripts/flowmatching/train_crpo_A_from_FT_expert.sh
set -e
cd "$(dirname "$0")/../.."

exp_id="CRPO_A_FTexpert_CLAP"
model="fluxaudio_score_s_v2_44k"
weights="exps/FT_expert_only_r1/FT_expert_only_r1_last.pth"
crpo_pair_dir="data/crpo_pairs/FT_expert_CRPO_A_CLAP"
beta=${CRPO_BETA:-2000.0}
lambda_fm=${CRPO_LAMBDA_FM:-1.0}
lr=${LR:-1e-6}
iters=${ITERS:-5000}

echo "[CRPO-A] from FT_expert_only_r1, beta=$beta lambda_fm=$lambda_fm lr=$lr iters=$iters"
echo "[CRPO-A] pair dir: $crpo_pair_dir"

torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) \
    train.py --config-name train_config_crpo \
    exp_id=$exp_id compile=False model=$model \
    weights=$weights \
    use_score=True use_crpo=True \
    null_score_prob=0.1 \
    crpo_beta=$beta crpo_lambda_fm=$lambda_fm \
    learning_rate=$lr num_iterations=$iters \
    batch_size=8 \
    +data.crpo_pair_dir=$crpo_pair_dir \
    +use_rope=True \
    +use_wandb=False
