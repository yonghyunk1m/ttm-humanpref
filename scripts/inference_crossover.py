"""Inference for ICME 2026 ATTM.

Generates audio with one (or, optionally, two crossover-mixed) score-conditioned
FluxAudio checkpoint(s) over the Euler ODE steps, with an optional per-step
CFG schedule (constant / interval / linear anneal). The submitted pipeline uses
the single-checkpoint mode; the optional ``slot2_*`` arguments are kept for
ablation studies that mix two checkpoints across the denoising trajectory.

Example (single-checkpoint, submitted pipeline):
  CUDA_VISIBLE_DEVICES=0 python scripts/inference_crossover.py \\
      --slot1_variant fluxaudio_score_s_v2_44k \\
      --slot1_weights weights/sub1_seed42_fluxaudio_s.pt \\
      --prompts_csv data/sdd100_prompts.csv \\
      --score1 5.0 \\
      --cfg_mode constant --cfg_hi 4.0 \\
      --num_steps 25 \\
      --output output/sub1_raw

Notes:
  - Step indexing uses t in [1, 0] (reverse_flow=True). ``--handoff`` (only
    relevant in two-checkpoint crossover mode) is the normalized progress
    (1 - t) at which the second checkpoint takes over from the first.
  - Both models must share latent shape and use the same v2_44k seq config.
  - Outputs raw 44.1kHz wavs.
"""
import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# PyTorch <2.6 CVE workaround, same as inference_rejection.py
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.networks_score import SCORE_MODEL_REGISTRY
from meanaudio.model.utils.features_utils import FeaturesUtils

TARGET_SR = 44100
TARGET_LEN = TARGET_SR * 10
DEFAULT_NEG = ("noise, distortion, low quality, static, hum, hiss, clipping, "
               "muffled, amateur recording")


def load_model(variant, weights, device):
    lm = torch.load('sets/latent_mean_44k.pt', map_location='cpu')
    ls = torch.load('sets/latent_std_44k.pt', map_location='cpu')
    empty_t5 = torch.load('weights/empty_string_t5.pth', map_location='cpu')
    empty_c = torch.load('weights/empty_string_clap_c.pth', map_location='cpu')
    model = SCORE_MODEL_REGISTRY[variant](
        latent_mean=lm, latent_std=ls,
        empty_string_feat=empty_t5, empty_string_feat_c=empty_c,
        use_rope=True, text_c_dim=512, null_score_prob=0.1,
    ).to(device).eval()
    sd = torch.load(weights, map_location='cpu', weights_only=True)
    for k in ('empty_string_feat', 'empty_string_feat_c'):
        if k in sd and sd[k].shape != model.state_dict()[k].shape:
            sd[k] = sd[k].view(model.state_dict()[k].shape)
    model.load_state_dict(sd, strict=False)
    return model


def build_cfg_schedule(args):
    """Returns fn(progress in [0,1]) -> cfg_strength."""
    mode = args.cfg_mode
    if mode == 'constant':
        val = args.cfg_hi
        return lambda p: val
    if mode == 'anneal':
        hi, lo = args.cfg_hi, args.cfg_lo
        return lambda p: hi + (lo - hi) * p
    if mode == 'interval':
        hi, lo = args.cfg_hi, args.cfg_lo
        a, b = args.cfg_interval
        return lambda p: hi if a <= p <= b else lo
    raise ValueError(mode)


def load_prompts(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({'id': r['id'], 'prompt': r['prompt']})
    return rows


def prep_conditions(model, feats, prompt, negative):
    tf, tfc = feats.encode_text([prompt])
    cond = model.preprocess_conditions(tf, tfc)
    if negative:
        nf, nfc = feats.encode_text([negative])
        neg = model.preprocess_conditions(nf, nfc)
    else:
        empty_tf = model.empty_string_feat
        while empty_tf.dim() > 3:
            empty_tf = empty_tf.squeeze(0)
        empty_tfc = model.empty_string_feat_c
        while empty_tfc.dim() > 1:
            empty_tfc = empty_tfc.squeeze(0)
        neg = model.preprocess_conditions(
            empty_tf.expand(1, -1, -1), empty_tfc.expand(1, -1))
    return cond, neg


def generate_one(args, model1, model2, feats, prompt, seed, cfg_fn, device):
    cond1, neg1 = prep_conditions(model1, feats, prompt, args.negative)
    if model2 is not None:
        cond2, neg2 = prep_conditions(model2, feats, prompt, args.negative)
    else:
        cond2 = neg2 = None

    rng = torch.Generator(device=device).manual_seed(seed)
    latent_seq_len = getattr(model1, 'latent_seq_len', 430)
    latent_dim = getattr(model1, 'latent_dim', 40)
    x = torch.randn(1, latent_seq_len, latent_dim, device=device, generator=rng)

    # reverse flow: t goes 1 → 0
    steps = torch.linspace(1.0, 0.0, args.num_steps + 1, device=device)
    handoff_step = int(round(args.handoff * args.num_steps)) if model2 is not None else args.num_steps

    for i in range(args.num_steps):
        t = steps[i]
        progress = 1.0 - t.item()
        cfg = cfg_fn(progress)

        # Which model runs this step?
        if args.mode == 'single' or model2 is None:
            m, s_val, c_pos, c_neg = model1, args.score1, cond1, neg1
        elif args.mode == 'crossover':
            # Slot 2 early (noise→content), Slot 1 late (content→detail)
            if i < handoff_step:
                m, s_val, c_pos, c_neg = model2, args.score2, cond2, neg2
            else:
                m, s_val, c_pos, c_neg = model1, args.score1, cond1, neg1
        elif args.mode == 'crossover_reverse':
            if i < handoff_step:
                m, s_val, c_pos, c_neg = model1, args.score1, cond1, neg1
            else:
                m, s_val, c_pos, c_neg = model2, args.score2, cond2, neg2
        elif args.mode == 'avg':
            # Pure velocity averaging every step
            m = None
        else:
            raise ValueError(args.mode)

        t_b = t * torch.ones(1, device=device)

        if args.mode == 'avg':
            s1 = torch.full((1,), args.score1, device=device)
            s2 = torch.full((1,), args.score2, device=device)
            v1 = model1.predict_flow(x, t_b, cond1, s1)
            v2 = model2.predict_flow(x, t_b, cond2, s2)
            null = model1.get_empty_score(1, device)
            v1n = model1.predict_flow(x, t_b, neg1, null)
            null2 = model2.get_empty_score(1, device)
            v2n = model2.predict_flow(x, t_b, neg2, null2)
            w = args.avg_weight
            v_cond = w * v1 + (1 - w) * v2
            v_neg = w * v1n + (1 - w) * v2n
        else:
            s = torch.full((1,), s_val, device=device)
            v_cond = m.predict_flow(x, t_b, c_pos, s)
            null = m.get_empty_score(1, device)
            v_neg = m.predict_flow(x, t_b, c_neg, null)

        v = cfg * v_cond + (1 - cfg) * v_neg
        dt = steps[i + 1] - t
        if args.solver == 'heun' and i < args.num_steps - 1:
            # Predict endpoint, evaluate velocity there, trapezoid-average.
            x_pred = x + dt * v
            t_next = steps[i + 1] * torch.ones(1, device=device)
            # Re-evaluate v using the same (m, s_val, c_pos, c_neg) as above
            if args.mode == 'avg':
                s1 = torch.full((1,), args.score1, device=device)
                s2 = torch.full((1,), args.score2, device=device)
                v1 = model1.predict_flow(x_pred, t_next, cond1, s1)
                v2 = model2.predict_flow(x_pred, t_next, cond2, s2)
                null = model1.get_empty_score(1, device)
                v1n = model1.predict_flow(x_pred, t_next, neg1, null)
                null2 = model2.get_empty_score(1, device)
                v2n = model2.predict_flow(x_pred, t_next, neg2, null2)
                v_cond_2 = args.avg_weight * v1 + (1 - args.avg_weight) * v2
                v_neg_2 = args.avg_weight * v1n + (1 - args.avg_weight) * v2n
            else:
                s = torch.full((1,), s_val, device=device)
                v_cond_2 = m.predict_flow(x_pred, t_next, c_pos, s)
                null = m.get_empty_score(1, device)
                v_neg_2 = m.predict_flow(x_pred, t_next, c_neg, null)
            # Use same cfg as current step for simplicity (could interpolate)
            v2_mixed = cfg * v_cond_2 + (1 - cfg) * v_neg_2
            x = x + dt * 0.5 * (v + v2_mixed)
        else:
            x = x + dt * v

    x = model1.unnormalize(x)
    mel = feats.decode(x)
    audio = feats.vocode(mel).cpu().squeeze()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[-1] > TARGET_LEN:
        audio = audio[..., :TARGET_LEN]
    elif audio.shape[-1] < TARGET_LEN:
        audio = torch.nn.functional.pad(audio, (0, TARGET_LEN - audio.shape[-1]))
    return audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--slot1_variant', required=True)
    p.add_argument('--slot1_weights', required=True)
    p.add_argument('--slot2_variant', default=None)
    p.add_argument('--slot2_weights', default=None)
    p.add_argument('--prompts_csv', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--score1', type=float, default=4.5)
    p.add_argument('--score2', type=float, default=3.5)
    p.add_argument('--cfg_mode', default='constant',
                   choices=['constant', 'anneal', 'interval'])
    p.add_argument('--cfg_hi', type=float, default=2.5)
    p.add_argument('--cfg_lo', type=float, default=1.0)
    p.add_argument('--cfg_interval', type=float, nargs=2, default=[0.2, 0.8])
    p.add_argument('--mode', default='single',
                   choices=['single', 'crossover', 'crossover_reverse', 'avg'])
    p.add_argument('--handoff', type=float, default=0.5,
                   help='Progress in [0,1] at which to hand off Slot2→Slot1.')
    p.add_argument('--avg_weight', type=float, default=0.5,
                   help='Weight for model1 when mode=avg (model2 gets 1-w).')
    p.add_argument('--num_steps', type=int, default=25)
    p.add_argument('--seed_base', type=int, default=42)
    p.add_argument('--negative', default=DEFAULT_NEG)
    p.add_argument('--solver', default='euler', choices=['euler', 'heun'])
    p.add_argument('--prompt_prefix', default='',
                   help='String prepended to every prompt before encoding.')
    p.add_argument('--prompt_suffix', default='',
                   help='String appended to every prompt before encoding.')
    args = p.parse_args()

    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f'[crossover] loading Slot 1: {args.slot1_weights}', flush=True)
    model1 = load_model(args.slot1_variant, args.slot1_weights, device)
    if args.slot2_weights:
        print(f'[crossover] loading Slot 2: {args.slot2_weights}', flush=True)
        model2 = load_model(args.slot2_variant or args.slot1_variant,
                            args.slot2_weights, device)
    else:
        model2 = None

    feats = FeaturesUtils(tod_vae_ckpt='./weights/v1-44.pth',
                          enable_conditions=True, encoder_name='t5_clap',
                          mode='44k', need_vae_encoder=False).to(device).eval()

    prompts = load_prompts(args.prompts_csv)
    if args.prompt_prefix or args.prompt_suffix:
        for row in prompts:
            row['prompt'] = args.prompt_prefix + row['prompt'] + args.prompt_suffix
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_fn = build_cfg_schedule(args)
    fm = FlowMatching(num_steps=args.num_steps, reverse_flow=True,
                      inference_mode='euler')  # step loop here, fm unused
    del fm

    start = time.time()
    with torch.no_grad():
        for i, row in enumerate(tqdm(prompts, desc='crossover')):
            out_path = out_dir / f"{row['id']}.wav"
            if out_path.exists():
                continue
            seed = args.seed_base + i
            audio = generate_one(args, model1, model2, feats, row['prompt'],
                                 seed, cfg_fn, device)
            torchaudio.save(str(out_path), audio, TARGET_SR)
            torch.cuda.empty_cache()
    elapsed = time.time() - start
    print(f'[crossover] wrote {len(prompts)} wavs to {out_dir} in {elapsed:.1f}s',
          flush=True)


if __name__ == '__main__':
    main()
