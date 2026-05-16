"""Quick A/B eval: CLAP cosine + FAD (SDD GT) on a list of audio dirs.
Runs 3×mdx (demucs) before CLAP/FAD to match the Slot 1/2 baseline pipeline.
"""
import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torchaudio
from scipy import linalg
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

DRYRUN_CSV_DEFAULT = 'data/dryrun/dry-run_prompts.csv'
GT_CACHE = '/tmp/sdd_gt_filelist_cache.npy'
CLAP_CKPT = 'weights/music_audioset_epoch_15_esc_90.14.pt'


def compute_fd(pred, gt):
    m1, s1 = gt.mean(0), np.cov(gt, rowvar=False)
    m2, s2 = pred.mean(0), np.cov(pred, rowvar=False)
    d = m1 - m2
    cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(d.dot(d) + np.trace(s1) + np.trace(s2) - 2 * np.trace(cm))


def run_3xmdx(src_dir, dst_dir, mdx_model):
    from demucs.apply import apply_model
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    cur = src_dir
    for p in range(3):
        nxt = Path(str(dst_dir) + f'_mdx{p + 1}')
        nxt.mkdir(parents=True, exist_ok=True)
        for wav_path in sorted(cur.glob('*.wav')):
            out = nxt / wav_path.name
            if out.exists():
                continue
            wav, sr = torchaudio.load(wav_path)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            with torch.no_grad():
                sources = apply_model(mdx_model, wav.unsqueeze(0).cuda())
            instr = sources[0, :3].sum(0).mean(0, keepdim=True).cpu()
            instr = instr / (instr.abs().max() + 1e-8) * 0.95
            torchaudio.save(str(out), instr, sr)
            torch.cuda.empty_cache()
        cur = nxt
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True,
                    help='List of audio dirs to evaluate.')
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--skip_mdx', action='store_true')
    ap.add_argument('--mdx_root', default='/tmp/crossover_mdx')
    ap.add_argument('--out_json', default='output/crossover_eval/results.json')
    ap.add_argument('--prompts_csv', default=DRYRUN_CSV_DEFAULT)
    args = ap.parse_args()

    labels = args.labels or [Path(d).name for d in args.dirs]
    assert len(labels) == len(args.dirs)

    device = 'cuda'

    # Load prompts
    prompts = {}
    with open(args.prompts_csv) as f:
        for r in csv.DictReader(f):
            prompts[int(r['id'])] = r['prompt']

    # Load CLAP
    import laion_clap
    torch.backends.cuda.matmul.allow_tf32 = True
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': kw.get('weights_only', False)})
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap.load_ckpt(ckpt=CLAP_CKPT)
    torch.load = _orig
    clap.cuda().eval()

    text_dict = {pid: clap.get_text_embedding([p], use_tensor=False)[0]
                 for pid, p in prompts.items()}

    gt_mat = np.load(GT_CACHE)
    print(f'[eval] GT matrix: {gt_mat.shape}', flush=True)

    if not args.skip_mdx:
        from demucs.pretrained import get_model
        mdx = get_model('mdx_extra').eval().cuda()
    else:
        mdx = None

    import json
    results = {}
    for lbl, d in zip(labels, args.dirs):
        d = Path(d)
        if not d.is_dir():
            print(f'[eval] skip {lbl}: {d} missing', flush=True)
            continue
        # Apply 3×mdx
        if not args.skip_mdx:
            mdx_out = Path(args.mdx_root) / lbl
            final = run_3xmdx(d, mdx_out, mdx)
        else:
            final = d
        wavs = sorted(final.glob('*.wav'))
        print(f'[eval] {lbl}: {len(wavs)} wavs from {final}', flush=True)
        audio_emb = {}
        for f in tqdm(wavs, desc=lbl[:40], leave=False):
            emb = clap.get_audio_embedding_from_filelist(x=[str(f)], use_tensor=False)
            audio_emb[int(f.stem)] = emb[0]
            torch.cuda.empty_cache()
        pred_mat = np.stack([audio_emb[k] for k in sorted(audio_emb.keys())])
        fad = compute_fd(pred_mat, gt_mat)
        sims = [
            np.dot(text_dict[pid], audio_emb[pid]) /
            (np.linalg.norm(text_dict[pid]) *
             np.linalg.norm(audio_emb[pid]) + 1e-8)
            for pid in sorted(text_dict.keys()) if pid in audio_emb
        ]
        clap_score = float(np.mean(sims))
        print(f'[eval] {lbl}: FAD {fad:.4f} | CLAP {clap_score:.4f}',
              flush=True)
        results[lbl] = {'fad': fad, 'clap': clap_score, 'n': len(wavs)}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[eval] wrote {args.out_json}', flush=True)

    # Pretty table
    print()
    print(f'{"label":40} {"FAD":>8} {"CLAP":>8}')
    print('-' * 60)
    for lbl, r in results.items():
        print(f'{lbl:40} {r["fad"]:8.4f} {r["clap"]:8.4f}')


if __name__ == '__main__':
    main()
