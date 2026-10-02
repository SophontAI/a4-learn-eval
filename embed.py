# Step 3 of 4. Turns every prepared scan (prepare.py) into fixed-length vectors ("embeddings") with each foundation
# model in MODELS. The walnut models are frozen ViT encoders trained as masked autoencoders on 3D brain MRI
# (hf://medarc/walnut, model code from github.com/MedARC-AI/smri-fm). Each scan is given to the encoder the way its
# pretraining scans were: centre-padded to 208 x 240 x 208 and z-scored inside the brain, zero outside.
#
# The encoder returns one token (vector) per 8 x 8 x 8 mm patch of the brain. The eval needs one vector per scan, so
# the tokens are averaged ("pooled") in three fixed ways, each a separate representation in eval.py:
#   global   mean over all brain tokens
#   mtl      weighted mean over the medial temporal lobe (hippocampus, amygdala, inferior horn of the lateral
#            ventricle), each token weighted by the share of its patch inside those structures; this is where
#            Alzheimer's atrophy starts
#   regions  the same weighted mean within each of eight SynthSeg regions, concatenated (8 x the model's width)
#
# Only the forward pass runs here: about 4 minutes for all 5,745 scans and three models on 8 GPUs. To evaluate a new
# checkpoint, add it to MODELS (the model code in smri-fm must know its architecture), run
#   sbatch -p n --qos=high --account=sophont --gres=gpu:1 -c 16 --mem=96G --array=0-7 -o /data/paul/a4/mri/logs/embed_%A_%a.log \
#     --wrap 'set -e; pids=""; for j in 0 1 2; do SHARD=$((SLURM_ARRAY_TASK_ID * 3 + j)) uv run --locked python embed.py &
#             pids="$pids $!"; done; for pid in $pids; do wait "$pid"; done'
# and add its name to MODELS in eval.py. Writes /data/paul/a4/mri/embed/<model>_<pool>/shard<k>.parquet (index
# BID, session; float32 columns e0..; in "regions" the eight regions are consecutive blocks in REGIONS order) and,
# from shard 0, /data/paul/a4/mri/embed/models.json (parameter count, width and seconds per scan of each model).
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from fomo_tune.backbone import load_backbone

# checkpoints of hf://medarc/walnut, read from the cluster's shared Hugging Face cache
CKPT = "/data/smri-datasets/huggingface/hub/models--medarc--walnut/snapshots/fc2ef91fc5c1d327dd251f710b8a3fb6f48923a5/checkpoints/"
MODELS = {"walnut-v0.1-vitl": CKPT + "walnut-v0-1/vitl/sub-52k/checkpoint-last.pth",  # ViT-Large, best on the team benchmark
          "walnut-fomo300-vitl": CKPT + "pretrain_full_90_10_h100/checkpoint-last.pth",  # ViT-Large, the FOMO26 default
          "walnut-v0.1-vitb": CKPT + "walnut-v0-1/vitb/sub-52k/checkpoint-last.pth"}      # ViT-Base
PREPARED = "/data/paul/a4/mri/prepared/"
QC = "/data/paul/a4/mri/synthseg/"  # prepare.py's registration QC (dice_mni)
OUT = "/data/paul/a4/mri/embed/"
N_SHARDS = 24
SHAPE = (208, 240, 208)  # pretraining volume size: 26 x 30 x 26 patches of 8 mm
REGIONS = {"mtl": [5, 17, 18, 44, 53, 54],  # SynthSeg label codes; "mtl" is its own pool, the other eight make "regions"
           "hippocampus": [17, 53], "amygdala": [18, 54], "temporal_horn": [5, 44], "ventricles": [4, 43, 14, 15],
           "cortex": [3, 42], "white_matter": [2, 41], "deep_gray": [10, 11, 12, 13, 26, 28, 49, 50, 51, 52, 58, 60],
           "cerebellum_brainstem": [7, 8, 46, 47, 16]}

shard = int(os.environ["SHARD"])
files = sorted(glob.glob(PREPARED + "*.npz"))
assert len(files) == 5745, len(files)
files = files[shard::N_SHARDS]
dice = pd.read_parquet(QC).dice_mni
dev = torch.device("cuda")
models = {name: load_backbone(path)[0].to(dev).eval().requires_grad_(False) for name, path in MODELS.items()}
in_region = torch.zeros(len(REGIONS), 256, device=dev)  # lookup: SynthSeg code -> 1 if inside the region
for r, codes in enumerate(REGIONS.values()):
    in_region[r, codes] = 1
print(f"shard {shard}/{N_SHARDS}: {len(files)} scans; "
      + ", ".join(f"{name} {sum(p.numel() for p in m.parameters()) / 1e6:.0f}M parameters" for name, m in models.items()), flush=True)

vectors, index, seconds = {(name, pool): [] for name in MODELS for pool in ("global", "mtl", "regions")}, [], dict.fromkeys(MODELS, 0.0)
for i, f in enumerate(files):
    key = tuple(os.path.basename(f).removesuffix(".npz").split("_"))  # (BID, session)
    scan = dict(np.load(f))  # decompress each array once
    lo = [(n - k) // 2 for k, n in zip(scan["image"].shape, SHAPE)]  # centre padding, as in pretraining
    pad = [(a, n - k - a) for a, k, n in zip(lo, scan["image"].shape, SHAPE)]
    x = torch.from_numpy(np.pad(scan["image"], pad)).to(dev).float()
    labels = torch.from_numpy(np.pad(scan["labels"], pad)).to(dev).long()
    brain = labels > 0
    x = torch.where(brain, (x - x[brain].mean()) / x[brain].std(correction=0), 0)  # population SD, as in pretraining
    # share of each 8 mm patch inside each region: (regions, patches), patches in (x, y, z) order
    share = in_region[:, labels].reshape(len(REGIONS), 26, 8, 30, 8, 26, 8).mean((2, 4, 6)).reshape(len(REGIONS), -1)
    batch = {"image": x[None, None], "mask": brain[None, None], "affine": torch.eye(4, device=dev)[None]}
    for name, model in models.items():
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            out = model(batch)
        tokens, kept = out["patch_embeds"][0].float(), out["token_mask"][0].bool()
        weight = share[:, out["patch_ids"][0]] * kept
        # an empty region is only allowed in scans whose registration failed (eval.py drops those)
        assert (weight.sum(1) > 0).all() or dice[key] < 0.9, f"{f}: a region has no brain tokens"
        pooled = (weight @ tokens) / weight.sum(1, keepdim=True)
        vectors[(name, "global")].append(tokens[kept].mean(0).cpu().numpy())
        vectors[(name, "mtl")].append(pooled[0].cpu().numpy())
        vectors[(name, "regions")].append(pooled[1:].reshape(-1).cpu().numpy())
        seconds[name] += time.time() - t0  # .cpu() waited for the GPU
    index.append(key)
    if i % 50 == 0:
        print(f"{i}/{len(files)} {key}", flush=True)

index = pd.MultiIndex.from_tuples(index, names=["BID", "session"])
assert index.is_unique
for (name, pool), v in vectors.items():
    os.makedirs(OUT + f"{name}_{pool}", exist_ok=True)
    pd.DataFrame(np.stack(v), index=index, columns=[f"e{j}" for j in range(len(v[0]))]).to_parquet(OUT + f"{name}_{pool}/shard{shard}.parquet")
if shard == 0:
    json.dump({name: dict(parameters=sum(p.numel() for p in m.parameters()), width=len(vectors[(name, "global")][0]),
                          seconds_per_scan=seconds[name] / len(files)) for name, m in models.items()}, open(OUT + "models.json", "w"), indent=1)
print(f"shard {shard} done; forward seconds per scan " + ", ".join(f"{name} {s / len(files):.2f}" for name, s in seconds.items()), flush=True)
