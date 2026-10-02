# Step 4 of 4. The eval: scores every representation (one vector per MRI scan; see embed.py) on what a preclinical
# Alzheimer's trial needs from a scan, always against conventional volumetrics on the same scans. Version 1, frozen
# 2026-10-01; README.md explains the tasks, the rules and how to read the results. Tasks:
#   P1   PACC decline: each A4 participant's rate of cognitive decline (PACC points per year). Headline = R2 gained over
#        a model of baseline clinical scores + plasma p-tau217 (+ treatment arm). Also scored from the scan alone.
#   P1t  P1 within the tau PET substudy, gained over clinical + p-tau217 + tau PET: does the scan add to tau PET?
#   P2   PACC change at week 240 (completers; the target used by Devanarayan et al. 2025): R2 gained over clinical.
#   P3   CDR progression: worsening on the Clinical Dementia Rating by week 240: AUROC gained over clinical.
#   B1-3 tau PET, amyloid PET (centiloid), plasma p-tau217 predicted from the scan alone (R2).
#   S1-2 age and NeuroQuant hippocampal volume predicted from the scan alone (R2): sanity checks.
#   C1   A4-LEARN change: how well week-240 minus baseline change separates the cohorts after linear age adjustment
#        (d, in A4 standard deviations). This observational contrast does not isolate a disease or treatment effect.
#   C2   whether that change index tracks PACC decline (Spearman rho). C3: its size relative to screening-to-week-12 variability.
# Run (about 6 minutes on 64 CPUs):
#   sbatch -p c --qos=high --account=sophont -c 64 --mem=256G -o /data/paul/a4/eval/eval_%j.log --wrap "uv run --locked python eval.py"
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "/data/paul/a4/eval/"  # data.py's tables; the eval writes its results here too
MRI = "/data/paul/a4/mri/"    # prepare.py's synthseg/ and embed.py's embed/<model>_<pool>/
SEED = 20261001
FOLDS, REPEATS = 5, 10   # 5-fold cross-validation on dev, repeated with 10 fixed shuffles
BOOT = 2000              # participant bootstrap draws for intervals
NULL_DRAWS = 24          # random representations run through every task, to measure chance
JOBS = 48                # parallel workers
SPLIT_SHA = "0150ea566c4d4d7bebf1e7c37ddb0f453d9f65c030f7300073a9ff700500d4b7"  # sha256 of splits.csv: the split cannot drift
# foundation models (embed.py's names) and the three fixed ways their patch tokens are pooled into one vector per scan
MODELS = {"walnut v0.1 ViT-L": "walnut-v0.1-vitl", "walnut FOMO300 ViT-L": "walnut-fomo300-vitl", "walnut v0.1 ViT-B": "walnut-v0.1-vitb"}
POOLS = {"whole brain": "global", "medial temporal": "mtl", "8 regions": "regions"}
REFERENCE = "SynthSeg volumes"  # free conventional volumetrics on the same scans: the bar a foundation model must clear
# representations scored on the test split. Add a new model here once, when it is final; until then it gets dev scores only.
FROZEN = ["NeuroQuant volumes", "SynthSeg hippocampus", "SynthSeg volumes",
          "walnut v0.1 ViT-L, whole brain", "walnut v0.1 ViT-L, medial temporal", "walnut v0.1 ViT-L, 8 regions",
          "walnut FOMO300 ViT-L, whole brain", "walnut FOMO300 ViT-L, medial temporal", "walnut FOMO300 ViT-L, 8 regions",
          "walnut v0.1 ViT-B, whole brain", "walnut v0.1 ViT-B, medial temporal", "walnut v0.1 ViT-B, 8 regions"]
DESIGN_PAIR = ("walnut v0.1 ViT-L, medial temporal", REFERENCE)  # fixed pair used to compare ways of defining the target
# PACC test-form effects (points relative to form A), estimated in the trial's primary analysis model fit to these data
# (A4/LEARN readout package, a4.py); subtracting them stops alternating forms from looking like change
FORM = {"A": 0.0, "B": -0.28039, "SC": -0.72735}
MDE_K = 1.959964 + 0.841621  # z(97.5%) + z(80%): detectable difference = 2.80 x SE (two-sided 5% test, 80% power)
ALPHAS = np.logspace(-2, 6, 41)  # ridge penalties, chosen per fit by efficient leave-one-out
RIDGE = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
CLINICAL = ["age", "female", "edu", "apoe4",                                  # demographics and APOE e4 carrier status
            "pacc_bl", "fcsrt96_bl", "lm_delayed_bl", "dsst_bl", "mmse_v6_bl",  # PACC and its four tests at baseline
            "cfisp_bl", "cfipt_bl", "cdrsb_bl", "adlpqsp_bl", "c3_bl",           # complaints, CDR-SB, daily function, Cogstate
            "log_ptau217"]                                                      # plasma p-tau217
TAU = ["tau_mubada_bl", "tau_metatemporal_bl"]  # tau PET summaries
start = time.time()

# ================= 1. The participants and the frozen dev/test split =================
participants = pd.read_parquet(DATA + "participants.parquet")
participants["log_ptau217"] = np.log(participants.ptau217_bl)
participants["hippocampus_icv"] = 100 * (participants.nq_LeftHippocampus + participants.nq_RightHippocampus) / participants.nq_IntraCranialVolume
participants["session"] = participants.cohort.map({"A4": "004", "LEARN": "006"})  # each cohort's baseline MRI session
synthseg = pd.read_parquet(MRI + "synthseg")  # one row per scan: SynthSeg volumes and registration QC
assert synthseg.index.is_unique and len(synthseg) == 5745
synthseg = synthseg[synthseg.dice_mni >= 0.9]  # drops the one failed registration (B69617870 week 240, Dice 0.35)
has_baseline_scan = pd.MultiIndex.from_arrays([participants.index, participants.session]).isin(synthseg.index)
# the eval sample: A4 participants in the modified intention-to-treat set, and LEARN, each with a usable baseline scan
# and NeuroQuant volumes (so every representation covers exactly the same people)
sample = participants[(participants.mitt | (participants.cohort == "LEARN")) & has_baseline_scan & participants.nq_IntraCranialVolume.notna()]
strata = sample.cohort + sample.arm + (sample.apoe4 == 1).astype(str) + sample.tau_substudy.astype(str)
dev, test = train_test_split(sample.index, test_size=0.25, random_state=SEED, stratify=strata)
folds = pd.DataFrame(-1, index=sample.index, columns=[f"fold{r}" for r in range(REPEATS)])  # -1 = test participant
for r in range(REPEATS):
    for k, (_, held_out) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + r).split(np.sort(dev), strata[np.sort(dev)])):
        folds.loc[np.sort(dev)[held_out], f"fold{r}"] = k
splits = folds.assign(split=np.where(sample.index.isin(test), "test", "dev"))
sha = hashlib.sha256(splits.to_csv().encode()).hexdigest()
Path(DATA).mkdir(parents=True, exist_ok=True)
splits.to_csv(DATA + "splits.csv")
assert sha == SPLIT_SHA, f"split hash {sha} != SPLIT_SHA: the participant set or the split changed"

# ================= 2. Representations: one table per representation, indexed by (BID, session) =================
nq = [c for c in participants.columns if c.startswith("nq_") and c not in ("nq_IntraCranialVolume", "nq_HOC")]
ss = [c for c in synthseg.columns if c.startswith(("L_", "R_")) or c in ("csf", "third_ventricle", "fourth_ventricle", "brainstem")]
representations = {
    # NeuroQuant regional volumes as % of intracranial volume, plus hippocampal occupancy and ICV (baseline scans only)
    "NeuroQuant volumes": (100 * sample[nq].div(sample.nq_IntraCranialVolume, axis=0))
                          .assign(hoc=sample.nq_HOC, icv=sample.nq_IntraCranialVolume)
                          .set_axis(pd.MultiIndex.from_arrays([sample.index, sample.session], names=["BID", "session"])),
    "SynthSeg hippocampus": (100 * (synthseg.L_hippocampus + synthseg.R_hippocampus) / synthseg.tiv).to_frame("hippocampus"),
    "SynthSeg volumes": (100 * synthseg[ss].div(synthseg.tiv, axis=0)).assign(tiv=synthseg.tiv),  # 32 structures, % of TIV
} | {f"{model}, {pool}": pd.read_parquet(MRI + f"embed/{name}_{p}").loc[lambda d: d.index.isin(synthseg.index)]
     for model, name in MODELS.items() for pool, p in POOLS.items()}
for k, table in representations.items():
    baseline = pd.MultiIndex.from_arrays([sample.index, sample.session])
    assert table.index.is_unique and baseline.isin(table.index).all() and np.isfinite(table.to_numpy()).all(), k

# ================= 3. Targets =================
a4 = sample[sample.cohort == "A4"]
# P1 target: each participant's least-squares slope of PACC (points per year) over every blinded-phase assessment,
# after removing test-form effects; only people with >= 4 assessments spanning >= 2 years
pacc = pd.read_parquet(DATA + "pacc.parquet")
pacc = pacc[(pacc.phase == "blinded") & pacc.BID.isin(a4.index)].dropna(subset=["pacc", "week", "form"])
assert pacc.form.isin(FORM).all()
pacc = pacc.assign(score=pacc.pacc - pacc.form.map(FORM), year=pacc.week / 52.1775).sort_values(["BID", "year"])
span = pacc.groupby("BID").year.agg(["size", "min", "max"])
pacc = pacc[pacc.BID.isin(span.index[(span["size"] >= 4) & (span["max"] - span["min"] >= 2)])]
slope = pacc.groupby("BID").apply(lambda d: np.polyfit(d.year, d.score, 1)[0], include_groups=False)
# reliability of that slope: slopes from alternate assessments of the same person should agree (split-half correlation)
halves = pacc.assign(half=pacc.groupby("BID").cumcount() % 2).groupby(["BID", "half"]).apply(
    lambda d: np.polyfit(d.year, d.score, 1)[0] if d.year.max() - d.year.min() >= 1 else np.nan, include_groups=False).unstack().dropna()
r_half = halves.corr().iloc[0, 1]
# P3 target: 1 = CDR progression event by week 252; 0 = no event and followed to week 228 or later; else unknown
progression = pd.Series(np.nan, index=a4.index)
progression[(a4.cdr_event == 1) & (a4.cdr_weeks <= 252)] = 1.0
progression[(a4.cdr_event == 0) & (a4.cdr_weeks >= 228)] = 0.0
complete = a4.index[a4[CLINICAL].notna().all(1)]  # prognostic tasks need every clinical baseline score
TASKS = {  # name -> (target, baseline block the scan must add to, metric, description)
    "P1": (slope.reindex(complete).dropna(), CLINICAL, "r2", "PACC decline rate (points/yr), every blinded visit, both arms"),
    "P1t": (slope.reindex(complete.intersection(a4.index[a4[TAU].notna().all(1)])).dropna(), CLINICAL + TAU, "r2",
            "PACC decline rate, tau PET substudy, on top of tau PET"),
    "P2": (a4.pacc_chg240.reindex(complete).dropna(), CLINICAL, "r2", "PACC change at week 240, completers, both arms"),
    "P3": (progression.reindex(complete).dropna(), CLINICAL, "auc", "CDR-global progression by week 240"),
    "B1": (a4.tau_metatemporal_bl.dropna(), [], "r2", "Tau PET temporal meta-ROI SUVR"),
    "B2": (a4.centiloid.dropna(), [], "r2", "Amyloid PET centiloid"),
    "B3": (a4.log_ptau217.dropna(), [], "r2", "Plasma p-tau217 (log)"),
    "S1": (sample.age, [], "r2", "Age (A4 and LEARN)"),
    "S2": (sample.hippocampus_icv, [], "r2", "NeuroQuant hippocampal volume, % ICV (A4 and LEARN)"),
}
print(f"[{time.time() - start:.0f}s] {len(sample)} participants ({(splits.split == 'dev').sum()} dev), split {sha[:12]}; "
      + ", ".join(f"{k} n={len(v[0])}" for k, v in TASKS.items()), flush=True)


# ================= 4. Fitting the probes =================
def crossval(model, X, y, fold, score_test):
    """Out-of-fold predictions for the dev rows (fold >= 0; participants x repeats), always with the same folds, and,
    if score_test, predictions for the test rows (fold -1) from one fit on all of dev."""
    dev_rows, test_rows = np.flatnonzero(fold[:, 0] >= 0), np.flatnonzero(fold[:, 0] < 0)
    oof = np.zeros((len(dev_rows), REPEATS))
    for r in range(REPEATS):
        for k in range(FOLDS):
            held_out = fold[dev_rows, r] == k
            oof[held_out, r] = clone(model).fit(X[dev_rows[~held_out]], y[dev_rows[~held_out]]).predict(X[dev_rows[held_out]])
    return oof, clone(model).fit(X[dev_rows], y[dev_rows]).predict(X[test_rows]) if score_test else None


# Each job fits one probe. Modes: "base" = the task's baseline block (+ the representation) + treatment arm;
# "alone" = the representation only; "no_tau" = P1t's baseline block without tau PET, to show tau PET's own gain.
# Baseline block + representation are combined by stacking: one ridge per block, so a 1,024-d embedding cannot drown
# out fifteen clinical scores, and a linear regression on the blocks' cross-fitted predictions combines them.
# Null jobs run random 1,024-d "representations" through every task: their spread is the chance variation of a fitted
# probe, which a bootstrap of fixed predictions cannot see.
jobs = [(t, None, "base") for t in TASKS if TASKS[t][1]] + [("P1t", None, "no_tau")] + \
       [(t, rep, mode) for t in TASKS for rep in representations for mode in (["base", "alone"] if TASKS[t][1] else ["alone"])
        if not (t == "S2" and rep == "NeuroQuant volumes")]  # NeuroQuant contains the S2 target
null_jobs = [(t, draw, mode) for t in TASKS for mode in (["base", "alone"] if TASKS[t][1] else ["alone"]) for draw in range(NULL_DRAWS)]
calls = []
for task, rep, mode in jobs + null_jobs:
    y, base, kind, _ = TASKS[task]
    cols = [] if mode == "alone" else [c for c in base if mode == "base" or c not in TAU]
    if rep is None:
        scan = None
    elif isinstance(rep, int):  # a null draw
        scan = np.random.default_rng(SEED + 1000 + rep).normal(size=(len(y), 1024))
    else:
        scan = representations[rep].loc[pd.MultiIndex.from_arrays([y.index, sample.session[y.index]])].to_numpy(float)
    blocks = ([sample.loc[y.index, cols].to_numpy(float)] if cols else []) + ([scan] if scan is not None else [])
    X = np.concatenate(blocks + ([sample.loc[y.index, ["treated"]].to_numpy(float)] if cols else []), axis=1)
    if len(blocks) == 2:
        n_base, n_scan = blocks[0].shape[1], blocks[1].shape[1]
        base_cols, scan_cols = list(range(n_base)) + [n_base + n_scan], list(range(n_base, n_base + n_scan))  # arm joins the baseline block
        model = StackingRegressor([(name, make_pipeline(ColumnTransformer([("cols", "passthrough", c)]), StandardScaler(), RidgeCV(alphas=ALPHAS)))
                                   for name, c in [("baseline", base_cols), ("scan", scan_cols)]], final_estimator=LinearRegression(), cv=5)
    else:
        model = RIDGE
    calls.append(delayed(crossval)(model, X, y.to_numpy(float), folds.loc[y.index].to_numpy(), rep is None or rep in FROZEN))
fits = Parallel(n_jobs=JOBS)(calls)
fits, null_fits = fits[:len(jobs)], fits[len(jobs):]
print(f"[{time.time() - start:.0f}s] {len(jobs)} probes and {len(null_jobs)} null probes done", flush=True)


# ================= 5. Scores and paired differences =================
def metric(kind, y, p):
    """R2 or AUROC (Mann-Whitney, ties averaged) along the last axis; leading axes broadcast (bootstrap draws, repeats)."""
    if kind == "r2":
        return 1 - ((y - p) ** 2).sum(-1) / ((y - y.mean(-1, keepdims=True)) ** 2).sum(-1)
    n1 = y.sum(-1)
    return ((stats.rankdata(p, axis=-1) * y).sum(-1) - n1 * (n1 + 1) / 2) / (n1 * (y.shape[-1] - n1))


# One set of bootstrap resamples per task, shared by every representation, so differences between them are paired.
# A dev score is the mean over the 10 cross-validation repeats; each bootstrap draw averages the repeats the same way.
rng = np.random.default_rng(SEED)
n_split = {t: [int((splits.split.loc[v[0].index] == s).sum()) for s in ("dev", "test")] for t, v in TASKS.items()}
draws = {t: (rng.integers(0, nd, (BOOT, nd)), rng.integers(0, nt, (BOOT, nt))) for t, (nd, nt) in n_split.items()}
scores, predictions = {}, []
for (task, rep, mode), (oof, test_pred) in zip(jobs, fits):
    y, _, kind, _ = TASKS[task]
    is_dev = (splits.split.loc[y.index] == "dev").to_numpy()
    y_dev, y_test = y.to_numpy()[is_dev], y.to_numpy()[~is_dev]
    boot_dev, boot_test = draws[task]
    per_repeat = metric(kind, y_dev[None], oof.T)
    row = dict(task=task, rep=rep, mode=mode, n_dev=len(y_dev), n_test=len(y_test), dev=per_repeat.mean(), dev_sd_repeats=per_repeat.std(),
               boot=metric(kind, y_dev[boot_dev][:, None], oof[boot_dev].transpose(0, 2, 1)).mean(1))
    row["dev_ci"] = np.quantile(row["boot"], [0.025, 0.975])
    if test_pred is not None:
        row |= dict(test=metric(kind, y_test, test_pred), test_boot=metric(kind, y_test[boot_test], test_pred[boot_test]))
        row["test_ci"] = np.quantile(row["test_boot"], [0.025, 0.975])
    scores[(task, rep, mode)] = row
    predictions.append(pd.DataFrame({"task": task, "rep": rep or "none", "mode": mode, "BID": np.r_[y.index[is_dev], y.index[~is_dev]],
                                     "split": ["dev"] * len(y_dev) + ["test"] * len(y_test), "y": np.r_[y_dev, y_test],
                                     "pred": np.r_[oof.mean(1), test_pred if test_pred is not None else np.full(len(y_test), np.nan)]}))
# gain = representation + baseline vs baseline alone; vs_reference = representation vs REFERENCE in the same mode;
# tau_gain = tau PET's own gain within P1t
for (task, rep, mode), row in scores.items():
    if rep:
        pairs = [("gain", (task, None, "base")), ("vs_reference", (task, REFERENCE, mode))]
    else:
        pairs = [("tau_gain", (task, None, "no_tau"))] if task == "P1t" and mode == "base" else []
    for key, other in pairs:
        if other not in scores or other == (task, rep, mode) or (key == "gain" and mode != "base"):
            continue
        diff = row["boot"] - scores[other]["boot"]
        row[key] = dict(dev=row["dev"] - scores[other]["dev"], dev_ci=np.quantile(diff, [0.025, 0.975]), dev_se=diff.std(), dev_p_le0=(diff <= 0).mean())
        if "test" in row and "test" in scores[other]:
            diff = row["test_boot"] - scores[other]["test_boot"]
            row[key] |= dict(test=row["test"] - scores[other]["test"], test_ci=np.quantile(diff, [0.025, 0.975]), test_se=diff.std())
chance = {}  # chance level of each headline: the scores of random representations
for (task, draw, mode), (oof, _) in zip(null_jobs, null_fits):
    y, _, kind, _ = TASKS[task]
    y_dev = y.to_numpy()[(splits.split.loc[y.index] == "dev").to_numpy()]
    chance.setdefault(f"{task} {mode}", []).append(metric(kind, y_dev[None], oof.T).mean() - (scores[(task, None, "base")]["dev"] if mode == "base" else 0))
print(f"[{time.time() - start:.0f}s] scoring done", flush=True)


# ================= 6. Cohort change (C1-C3): A4 vs LEARN =================
def change_index(X, X12, age, is_a4, fold, score_test):
    """Turns each participant's week-240 change vector into one number, cross-fitted. In each training fold: remove each
    feature's linear age effect (estimated net of cohort), then fit a ridge discriminant of A4 vs LEARN (a penalized
    linear discriminant) and rescale its direction to unit SD on the training data. Held-out week-240 change and week-12
    change (NaN without a week-12 scan) are projected onto that direction. Returns (participants x repeats) arrays; with
    score_test, one more fit on all of dev scores the test rows into column 0."""
    dev_rows, test_rows = np.flatnonzero(fold[:, 0] >= 0), np.flatnonzero(fold[:, 0] < 0)
    v, v12 = np.full((len(fold), REPEATS), np.nan), np.full((len(fold), REPEATS), np.nan)
    plan = [(r, k) for r in range(REPEATS) for k in range(FOLDS)] + ([(0, None)] if score_test else [])
    for r, k in plan:
        train, held_out = (dev_rows[fold[dev_rows, r] != k], dev_rows[fold[dev_rows, r] == k]) if k is not None else (dev_rows, test_rows)
        ageing = np.linalg.lstsq(np.c_[np.ones(len(train)), age[train], is_a4[train]], X[train], rcond=None)[0][1]
        discriminant = clone(RIDGE).fit(X[train] - age[train, None] * ageing, is_a4[train])
        w = discriminant[-1].coef_ / discriminant[0].scale_
        w /= np.std((X[train] - age[train, None] * ageing) @ w)  # without a fixed scale, folds' scales bias d toward zero
        v[held_out, r] = (X[held_out] - age[held_out, None] * ageing) @ w
        with_week12 = held_out[np.isfinite(X12[held_out]).all(1)]
        v12[with_week12, r] = X12[with_week12] @ w
    return v, v12


change_reps, calls, ids = [r for r in representations if r != "NeuroQuant volumes"], [], None  # NeuroQuant: baseline scans only
for rep in change_reps:
    table = representations[rep]
    baseline = table.loc[pd.MultiIndex.from_arrays([sample.index, sample.session])].set_axis(sample.index)
    change240 = (table.xs("066", level="session").reindex(sample.index) - baseline).dropna()
    assert ids is None or change240.index.equals(ids), rep  # the same participants for every representation (paired bootstrap)
    ids = change240.index
    change12 = (table.xs("009", level="session").reindex(ids) - baseline.loc[ids]).to_numpy(float)  # A4 only
    calls.append(delayed(change_index)(change240.to_numpy(float), change12, sample.age[ids].to_numpy(float),
                                       (sample.cohort[ids] == "A4").to_numpy(float), folds.loc[ids].to_numpy(), rep in FROZEN))
is_a4, age, decline = (sample.cohort[ids] == "A4").to_numpy(), sample.age[ids].to_numpy(), -slope.reindex(ids).to_numpy()
fold = folds.loc[ids].to_numpy()
change = {}
for rep, (v_all, v12_all) in zip(change_reps, Parallel(n_jobs=JOBS)(calls)):
    per_split = {}
    for name, rows in [("dev", np.flatnonzero(fold[:, 0] >= 0)), ("test", np.flatnonzero(fold[:, 0] < 0))]:
        if name == "test" and rep not in FROZEN:
            continue
        v, v12 = (v_all[rows], v12_all[rows]) if name == "dev" else (v_all[rows, :1], v12_all[rows, :1])
        stat = []  # row 0: the estimate; rows 1..: bootstrap draws (seeded alike for every representation, so paired)
        for i in [np.arange(len(rows))] + [np.random.default_rng(SEED + b).integers(0, len(rows), len(rows)) for b in range(BOOT // 2)]:
            a4_i, v_i, has_slope = is_a4[rows][i], v[i], np.isfinite(decline[rows][i]) & is_a4[rows][i]
            separation = np.linalg.lstsq(np.c_[np.ones(len(i)), a4_i, age[rows][i]], v_i, rcond=None)[0][1]  # age-adjusted A4 - LEARN
            rank_v, rank_decline = stats.rankdata(v_i[has_slope], axis=0), stats.rankdata(decline[rows][i][has_slope])
            rho = [np.corrcoef(rank_v[:, r], rank_decline)[0, 1] for r in range(v_i.shape[1])]
            stat.append([np.mean(separation / v_i[a4_i].std(0, ddof=1)), np.mean(rho), np.mean(separation / np.nanstd(v12[i][a4_i], 0, ddof=1))])
        stat = np.array(stat)
        per_split[name] = dict(n_a4=int(is_a4[rows].sum()), n_learn=int((~is_a4[rows]).sum()), n_week12=int(np.isfinite(v12[:, 0]).sum()),
                               d=stat[0, 0], d_ci=np.quantile(stat[1:, 0], [0.025, 0.975]),
                               rho_decline=stat[0, 1], rho_ci=np.quantile(stat[1:, 1], [0.025, 0.975]),
                               snr=stat[0, 2], snr_ci=np.quantile(stat[1:, 2], [0.025, 0.975]),
                               n_per_arm_25pct=2 * MDE_K ** 2 / (0.25 * stat[0, 0]) ** 2,  # hypothetical 25% contrast reduction, equal variance, no attrition
                               boot=stat[1:])
        if name == "dev":
            per_split[name]["d_sd_repeats"] = float(np.std([np.linalg.lstsq(np.c_[np.ones(len(rows)), is_a4[rows], age[rows]], v[:, r], rcond=None)[0][1]
                                                            / v[is_a4[rows], r].std(ddof=1) for r in range(REPEATS)]))
    change[rep] = dict(rep=rep, dims=representations[rep].shape[1]) | per_split
for rep, row in change.items():
    for name in ("dev", "test"):
        if rep != REFERENCE and name in row and name in change[REFERENCE]:
            diff = row[name]["boot"] - change[REFERENCE][name]["boot"]
            row[name]["vs_reference"] = {k: dict(diff=row[name][k] - change[REFERENCE][name][k],
                                                 ci=np.quantile(diff[:, j], [0.025, 0.975]), se=diff[:, j].std())
                                         for j, k in enumerate(["d", "rho_decline", "snr"])}
dev_c = np.flatnonzero(fold[:, 0] >= 0)  # chance level of C1-C2: random 1,024-d change vectors through the same pipeline
for v, _ in Parallel(n_jobs=JOBS)(delayed(change_index)(np.random.default_rng(SEED + 100 + i).normal(size=(len(ids), 1024)),
                                                        np.full((len(ids), 1024), np.nan), age.astype(float), is_a4.astype(float), fold, False)
                                  for i in range(NULL_DRAWS)):
    v_dev, a4_dev, has_slope = v[dev_c], is_a4[dev_c], is_a4[dev_c] & np.isfinite(decline[dev_c])
    separation = np.linalg.lstsq(np.c_[np.ones(len(dev_c)), a4_dev, age[dev_c]], v_dev, rcond=None)[0][1]
    chance.setdefault("C d", []).append(np.mean(separation / v_dev[a4_dev].std(0, ddof=1)))
    chance.setdefault("C rho_decline", []).append(np.mean([stats.spearmanr(v_dev[has_slope, r], decline[dev_c][has_slope]).statistic for r in range(REPEATS)]))
chance = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for k, v in chance.items()}
print(f"[{time.time() - start:.0f}s] change tasks done", flush=True)

# ================= 7. How good is the eval itself =================
# Detectable difference (80% power) = 2.80 x sqrt(SE^2 + 2 SD^2): SE = median paired-bootstrap SE of
# (representation - reference), SD = chance variation of one fitted probe (from the null draws), once per representation.
HEADLINE = {t: ("gain", "base") if TASKS[t][1] else ("dev", "alone") for t in TASKS}
meta = dict(split_sha=sha, n=dict(dev=int((splits.split == "dev").sum()), test=int((splits.split == "test").sum())),
            p1_split_half_r=r_half, p1_reliability=2 * r_half / (1 + r_half),  # Spearman-Brown: reliability of the full slope
            p1_slope_mean=TASKS["P1"][0].mean(), p1_slope_sd=TASKS["P1"][0].std(), resolution={}, rank_transfer={})
for task in TASKS:
    for mode in (["base", "alone"] if TASKS[task][1] else ["alone"]):
        rows = [r for (t, rep, m), r in scores.items() if t == task and m == mode and rep is not None]
        se = [r["vs_reference"]["dev_se"] for r in rows if "vs_reference" in r]
        se_test = [r["vs_reference"]["test_se"] for r in rows if "vs_reference" in r and "test_se" in r["vs_reference"]]
        frozen = [r for r in rows if "test" in r]
        sd = chance[f"{task} {mode}"]["sd"]
        meta["resolution"][f"{task} {mode}"] = dict(mdd_bootstrap=MDE_K * np.median(se), mdd_dev=MDE_K * np.sqrt(np.median(se) ** 2 + 2 * sd ** 2),
                                                   mdd_test=MDE_K * np.sqrt(np.median(se_test) ** 2 + 2 * sd ** 2),
                                                   sd_repeats=np.median([r["dev_sd_repeats"] for r in rows]))
        # do dev and test rank the frozen representations alike?
        meta["rank_transfer"][f"{task} {mode}"] = stats.spearmanr([r["dev"] for r in frozen], [r["test"] for r in frozen]).statistic
for k, sd in [("d", chance["C d"]["sd"]), ("rho_decline", chance["C rho_decline"]["sd"]), ("snr", 0.0)]:
    se = np.median([r["dev"]["vs_reference"][k]["se"] for r in change.values() if "vs_reference" in r["dev"]])
    meta["resolution"][f"C {k}"] = dict(mdd_bootstrap=MDE_K * se, mdd_dev=MDE_K * np.sqrt(se ** 2 + 2 * sd ** 2))
# The same two representations under three ways of defining the prognostic target: which resolves their difference best?
design = {}
placebo = sample.index[sample.treated == 0]
all_predictions = pd.concat(predictions)
for name, task, keep in [("P2 scored on placebo (Devanarayan)", "P2", placebo), ("P2 scored on both arms", "P2", None), ("P1 slope, both arms", "P1", None)]:
    x = all_predictions.query("task == @task and split == 'dev' and mode == 'base' and rep in @DESIGN_PAIR")
    x = x.pivot(index="BID", columns="rep", values=["y", "pred"])
    x = x if keep is None else x[x.index.isin(keep)]
    y, pred_a, pred_b = x[("y", DESIGN_PAIR[0])].to_numpy(), x[("pred", DESIGN_PAIR[0])].to_numpy(), x[("pred", DESIGN_PAIR[1])].to_numpy()
    i = np.random.default_rng(SEED).integers(0, len(y), (BOOT, len(y)))
    diff = metric("r2", y[i], pred_a[i]) - metric("r2", y[i], pred_b[i])
    delta = metric("r2", y, pred_a) - metric("r2", y, pred_b)
    design[name] = dict(n=len(y), delta=delta, se=diff.std(), z=delta / diff.std(), mdd=MDE_K * diff.std())
meta["design"] = design
meta["null"] = chance | dict(draws=NULL_DRAWS, dims=1024)
print(f"[{time.time() - start:.0f}s] meta done", flush=True)

# ================= 8. Outputs =================
R = dict(tasks={k: dict(description=v[3], metric=v[2], base=v[1], n=len(v[0]), n_dev=n_split[k][0], n_test=n_split[k][1], headline=HEADLINE[k])
                for k, v in TASKS.items()},
         reps={k: dict(dims=v.shape[1], frozen=k in FROZEN) for k, v in representations.items()}, reference=REFERENCE, design_pair=DESIGN_PAIR,
         results=[{k: v for k, v in r.items() if k not in ("boot", "test_boot")} | dict(rep=r["rep"] or "none") for r in scores.values()],
         change=[{k: ({kk: vv for kk, vv in v.items() if kk != "boot"} if isinstance(v, dict) else v) for k, v in r.items()} for r in change.values()],
         meta=meta, config=dict(seed=SEED, folds=FOLDS, repeats=REPEATS, boot=BOOT, form_effects=FORM, clinical=CLINICAL, tau=TAU,
                                run_date=time.strftime("%Y-%m-%d"), runtime_s=time.time() - start))
js = re.sub(r"\bNaN\b", "null", json.dumps(R, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
open(DATA + "results.json", "w").write(js)  # aggregate statistics only
all_predictions.to_parquet(DATA + "predictions.parquet")  # participant-level: stays on /data
open(DATA + "report.html", "w").write(Path(__file__).with_name("report.html").read_text().replace("/*RESULTS*/", js))  # aggregate-only page
for (task, rep, mode), r in scores.items():
    print(f"{task:4s} {mode:6s} {(rep or 'none')[:38]:38s} dev {r['dev']:+.3f} [{r['dev_ci'][0]:+.3f},{r['dev_ci'][1]:+.3f}]"
          + (f" test {r['test']:+.3f}" if "test" in r else "")
          + "".join(f"  {key} {r[key]['dev']:+.3f} [{r[key]['dev_ci'][0]:+.3f},{r[key]['dev_ci'][1]:+.3f}]"
                    for key in ("gain", "vs_reference", "tau_gain") if key in r))
for rep, r in change.items():
    print(f"C    {rep[:38]:38s} d {r['dev']['d']:+.3f} [{r['dev']['d_ci'][0]:+.3f},{r['dev']['d_ci'][1]:+.3f}] rho {r['dev']['rho_decline']:+.3f} "
          f"SNR {r['dev']['snr']:.2f} n/arm {r['dev']['n_per_arm_25pct']:.0f}" + (f"  test d {r['test']['d']:+.3f}" if "test" in r else ""))
print(f"[{time.time() - start:.0f}s] wrote {DATA}results.json and report.html", flush=True)
