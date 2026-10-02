# Step 2 of 4. Prepares every A4 and LEARN T1-weighted MRI once, so that embedding a new model (embed.py) only needs
# a forward pass. The steps reproduce how the walnut foundation models' pretraining scans were prepared
# (smri-fm/src/smri_mae/build_fomo300k_sparse_wds.py):
#   1. SynthSeg 2.0 segments the scan into 32 brain structures (also giving each structure's volume and the
#      total intracranial volume, TIV) and so provides a brain mask.
#   2. ANTs registers the skull-stripped scan to the skull-stripped MNI152NLin2009cAsym 1 mm template with an affine
#      (12-parameter) transform, i.e. it standardizes position, size and orientation but not shape.
#   3. The scan (linear interpolation) and the SynthSeg labels (nearest neighbour) are resampled onto the template's
#      193 x 229 x 193 grid, intensities are set to zero outside the brain and divided by their brain mean (raw values
#      reach 1.4e5, beyond float16). Any further normalization is the encoder's job.
# Scans: A4 sessions 004 (screening, the baseline MRI), 009 (week 12), 027 (week 84), 048 (week 168), 066 (week 240);
# LEARN sessions 006 (baseline) and 066 (week 240). 5,745 scans in all.
#
# SynthSeg (about 20 s per scan, mostly CPU post-processing) and ANTs (about 14 s) dominate. SynthSeg's network needs
# 15-20 GB of GPU memory, so three shards share each GPU; 8 GPUs (24 shards) take about 2.5-4 hours:
#   sbatch -p n --qos=high --account=sophont --gres=gpu:1 -c 16 --mem=128G --array=0-7 -o /data/paul/a4/mri/logs/prepare_%A_%a.log \
#     --wrap 'set -e; pids=""; for j in 0 1 2; do SHARD=$((SLURM_ARRAY_TASK_ID * 3 + j)) ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=5 \
#             PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --locked python prepare.py &
#             pids="$pids $!"; done; for pid in $pids; do wait "$pid"; done'
# Writes (participant-level, keep under /data):
#   /data/paul/a4/mri/prepared/<BID>_<session>.npz   image (float16) and labels (uint8 SynthSeg codes) on the template grid
#   /data/paul/a4/mri/synthseg/shard<k>.parquet      per scan: structure volumes (mm^3), TIV, and registration QC:
#       dice_mni = overlap of the registered brain mask with the template's brain (eval.py drops scans below 0.9),
#       affine_scale = subject/template volume ratio
import glob
import os
import time

os.environ["ANTS_RANDOM_SEED"] = "20261001"  # ANTs samples image points at random; the seed keeps reruns close (threads still vary the last digits)
import ants  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from SynthSeg_pytorch import SynthSegPredictor  # noqa: E402

A4 = "/data/leema/a4/A4-bids/A4/"  # A4 T1w scans in BIDS layout: sub-<BID>/ses-<session>/anat/*_T1w.nii.gz
LEARN = "/data/paul/a4/learn_t1/"  # LEARN T1w scans: LEARN_MR_T1_<BID>_<session>.nii.gz (from R2 s3://medarc/leema/A4_downloads/Images/T1w/)
SESSIONS = ["004", "009", "027", "048", "066"]
TEMPLATE = "/data/paul/a4/template/MNI152NLin2009cAsym_res-01_T1w.nii.gz"  # from TemplateFlow (tpl-MNI152NLin2009cAsym)
OUT = "/data/paul/a4/mri/"
N_SHARDS = 24
LABELS = {2: "L_cerebral_wm", 3: "L_cortex", 4: "L_lateral_ventricle", 5: "L_inf_lat_ventricle", 7: "L_cerebellum_wm",
          8: "L_cerebellum_cortex", 10: "L_thalamus", 11: "L_caudate", 12: "L_putamen", 13: "L_pallidum", 14: "third_ventricle",
          15: "fourth_ventricle", 16: "brainstem", 17: "L_hippocampus", 18: "L_amygdala", 24: "csf", 26: "L_accumbens",
          28: "L_ventral_dc", 41: "R_cerebral_wm", 42: "R_cortex", 43: "R_lateral_ventricle", 44: "R_inf_lat_ventricle",
          46: "R_cerebellum_wm", 47: "R_cerebellum_cortex", 49: "R_thalamus", 50: "R_caudate", 51: "R_putamen", 52: "R_pallidum",
          53: "R_hippocampus", 54: "R_amygdala", 58: "R_accumbens", 60: "R_ventral_dc"}  # SynthSeg label codes

shard = int(os.environ["SHARD"])
files = sorted(f for s in SESSIONS for f in glob.glob(f"{A4}sub-*/ses-{s}/anat/*_T1w.nii.gz")) + sorted(glob.glob(LEARN + "LEARN_MR_T1_*.nii.gz"))
assert len(files) == 5745, len(files)
files = files[shard::N_SHARDS]
tmp = f"/dev/shm/a4prep_{os.environ['SLURM_JOB_ID']}_{shard}_"
synthseg = SynthSegPredictor(device="cuda")
assert [int(k) for k in synthseg.labels_segmentation] == [0, *LABELS]
template = ants.image_read(TEMPLATE)
seg, _, _, affine, _ = synthseg.segment(nib.load(TEMPLATE))  # the template's own brain mask, from the same segmenter
nib.save(nib.Nifti1Image(seg.astype(np.int16), affine), tmp + "template_seg.nii")
template_mask = ants.resample_image_to_target(ants.image_read(tmp + "template_seg.nii"), template, interp_type="nearestNeighbor") > 0
template_brain, template_mask_np = template * template_mask, template_mask.numpy() > 0
os.makedirs(OUT + "prepared", exist_ok=True)
print(f"shard {shard}/{N_SHARDS}: {len(files)} scans", flush=True)

qc = []
for i, f in enumerate(files):
    t0 = time.time()
    if f.startswith(A4):
        bid, session = f.split("/sub-")[1].split("/")[0], f.split("/ses-")[1][:3]
    else:
        bid, session = os.path.basename(f).removesuffix(".nii.gz").split("_")[-2:]
    seg, _, volumes, affine, _ = synthseg.segment(nib.load(f))
    nib.save(nib.Nifti1Image(seg.astype(np.int16), affine), tmp + "seg.nii")
    torch.cuda.empty_cache()  # hand SynthSeg's activations back before the next shard on this GPU needs them
    t1, labels = ants.image_read(f), ants.image_read(tmp + "seg.nii")
    t_seg = time.time()
    brain = t1 * (ants.resample_image_to_target(labels, t1, interp_type="nearestNeighbor") > 0)
    reg = ants.registration(fixed=template_brain, moving=brain, type_of_transform="Affine")
    x = ants.apply_transforms(fixed=template, moving=t1, transformlist=reg["fwdtransforms"], interpolator="linear").numpy()
    s = ants.apply_transforms(fixed=template, moving=labels, transformlist=reg["fwdtransforms"], interpolator="nearestNeighbor").numpy()
    s = s.round().astype(np.uint8)
    dice = 2 * ((s > 0) & template_mask_np).sum() / ((s > 0).sum() + template_mask_np.sum())
    # determinant of the affine = subject volume / template volume (ANTs maps template points into the subject)
    scale = np.linalg.det(np.array(ants.read_transform(reg["fwdtransforms"][0]).parameters[:9]).reshape(3, 3))
    image = np.where(s > 0, x / x[s > 0].mean(), 0).astype(np.float16)
    np.savez_compressed(OUT + f"prepared/{bid}_{session}.npz", image=image, labels=s)
    qc.append(dict(BID=bid, session=session, tiv=volumes[0], **dict(zip(LABELS.values(), volumes[1:])), dice_mni=dice,
                   affine_scale=scale, brain_voxels=int((s > 0).sum()), s_synthseg=t_seg - t0, s_register=time.time() - t_seg))
    if i % 25 == 0:
        print(f"{i}/{len(files)} {bid} ses-{session} dice {dice:.3f} scale {scale:.2f} "
              f"synthseg {t_seg - t0:.1f}s register {time.time() - t_seg:.1f}s", flush=True)

os.makedirs(OUT + "synthseg", exist_ok=True)
pd.DataFrame(qc).set_index(["BID", "session"]).to_parquet(OUT + f"synthseg/shard{shard}.parquet")
print(f"shard {shard} done", flush=True)
