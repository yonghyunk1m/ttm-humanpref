#!/usr/bin/env bash
# FluxAudio-S 44kHz + Score Conditioning v2 (InputAdd)
# Native 44kHz generation — latent_dim=40, seq_len=430
# Requires 44k NPZ data (extract_audio_latents_44k.py)
#
# This script trains the v2 (InputAdd) SFT used for the SFT-only v2
# cells in the cross-mechanism ablation (paper §IV-B, Table 3). The
# deployed submission chain uses the v1 SFT instead — see
# train_fluxaudio_s_44k_score_v1.sh.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
NUM_GPUS=1
btz=32
num_iterations=200_000
exp_id=Score_v2_44k
text_encoder_name=t5_clap
text_c_dim=512
model=fluxaudio_score_s_v2_44k
null_score_prob=0.1

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    train.py \
    --config-name train_config_jamendo_44k_score.yaml \
    exp_id=$exp_id compile=False model=$model \
    batch_size=${btz} eval_batch_size=1 \
    num_iterations=$num_iterations \
    text_encoder_name=$text_encoder_name \
    data_dim.text_c_dim=$text_c_dim \
    pin_memory=False num_workers=10 ac_oversample_rate=5 \
    use_score=True use_meanflow=False null_score_prob=$null_score_prob \
    cfg_strength=4.5 enable_grad_scaler=False \
    ++grad_accum_steps=2 ++score_column=instrumental_reward_score \
    ++use_rope=True ++use_wandb=True ++debug=False
