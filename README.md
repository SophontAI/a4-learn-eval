# A4/LEARN imaging eval

An internal benchmark for brain-MRI representations, built on the A4 Alzheimer's prevention trial. Give it a vector of
numbers per MRI scan (for example a foundation model's embedding) and it measures whether the scan tells a clinical
trial something useful: who will decline, the disease biology behind it, and change over time that is specific to
Alzheimer's. Every number is compared, on the same people, with standard volume measurements of the same scans, so
it answers the question that matters for our foundation models: do they beat what a free segmentation tool gives?

The latest results are on the [eval page](https://claude.ai/artifact/HKrSo3SoLKGr9GVczneX5q) (private, ask Paul for
access) and summarized [below](#current-results).

## The study in one minute

**A4** (Anti-Amyloid Treatment in Asymptomatic Alzheimer's) screened about 4,500 people aged 65-85 with normal memory
using amyloid PET. The 1,169 with elevated brain amyloid, an early sign of Alzheimer's disease, were randomized to the
antibody solanezumab or placebo and followed for 240 weeks (about 4.6 years). The drug did not slow decline. **LEARN**
followed 538 people who screened amyloid-negative, untreated, with the same tests, as a reference for normal ageing.

What was measured, and what the eval uses:

| Measure | What it is |
|---|---|
| PACC | Preclinical Alzheimer Cognitive Composite, the trial's main outcome: the average of four memory and attention tests, scaled so 0 is the average at baseline and lower is worse. Given every 6-12 months. Placebo participants lost about 1.1 points over 240 weeks. |
| CDR | Clinical Dementia Rating: 0 normal, 0.5 very mild impairment, 1 or more dementia. |
| Amyloid PET | Brain amyloid load at screening, in centiloids. |
| Tau PET | Tau tangles, the second Alzheimer's protein, which tracks symptoms more closely. About 375 A4 participants had it. |
| Plasma p-tau217 | A blood test for Alzheimer's pathology; the strongest single predictor of decline in A4. |
| T1-weighted MRI | Brain anatomy. A4: screening (session 004) and weeks 12, 84, 168 and 240 (sessions 009, 027, 048, 066). LEARN: baseline (006) and week 240 (066). 5,745 scans in total. |

## What the eval asks

Each task asks one question and reports one headline number. "Gain" means how much a scan adds on top of what a trial
already knows at baseline: age, sex, education, APOE e4 genotype, the PACC tests, everyday-function and
cognitive-complaint questionnaires, CDR, a computerized test battery, and plasma p-tau217 (the "clinical baseline").

| Task | Question | Who (participants) | Headline |
|---|---|---|---|
| **P1** | Does the scan predict how fast someone's PACC declines, beyond the clinical baseline? **The primary task.** | A4, both arms, 893 | R² gained |
| P1 scan | The same target from the scan alone | 893 | R² |
| P1t | Does the scan add to tau PET? (P1 within the tau PET substudy, with tau PET in the baseline) | 296 | R² gained |
| P2 | PACC change at week 240 (the target in the published A4 prognosis paper, Devanarayan et al. 2025) | 746 | R² gained |
| P3 | Who worsens on the CDR by week 240? | 804 | AUROC gained |
| B1-B3 | Tau PET, amyloid PET and plasma p-tau217, read from the scan alone | 374-1,118 | R² |
| S1-S2 | Age and hippocampal volume from the scan: sanity checks a good representation should pass | 1,651 | R² |
| C1 | Does change in the scan over 240 weeks separate amyloid-positive A4 from amyloid-negative LEARN, after removing normal ageing? | 1,089 with both scans | d |
| C2 | Does that change track PACC decline? | A4 of C1 | Spearman ρ |
| C3 | Is that change large compared with noise between scans taken 12 weeks apart? | A4 of C1 | ratio |

Why the PACC slope is the headline: the rate of decline over every visit (at least 4 visits over at least 2 years) is
measured more reliably than the change at one visit (split-half reliability 0.87) and is available for more people,
so it can tell representations apart with a third of the participants a week-240 comparison would need.

## How it keeps us honest

- **One frozen split.** Participants are split once into dev (75%) and test (25%). The split is checked against a
  hash in the code, so it cannot change by accident.
- **Develop on dev, look at test once.** Dev scores come from 5-fold cross-validation repeated 10 times with fixed
  folds. Test scores are only computed for representations listed in `FROZEN` in `eval.py`; add a model there when it
  is final.
- **Fixed probes.** Every representation gets the same model: ridge regression with a fixed penalty grid. When the
  clinical baseline is included, the baseline and the scan each get their own ridge and a linear regression combines
  them, so a 1,024-number embedding cannot drown out fifteen clinical scores.
- **Paired comparisons.** A participant bootstrap (2,000 draws) shared by all representations gives every difference
  a confidence interval.
- **Calibrated chance.** 24 random representations go through every task. Their average is the chance level, and
  their spread is the extra noise from fitting a probe, which the bootstrap cannot see. All intervals and detectable
  differences include it.
- **Known resolution.** For each task the eval reports the smallest difference from the reference it can detect with
  80% power.

## How to read the results

- Gains near the chance level are not real. Shaded cells on the eval page (▲ better, ▼ worse than SynthSeg volumes)
  are differences whose 95% interval excludes zero. With about 150 comparisons, a few will appear by chance, so look
  for patterns across related tasks.
- Compare a difference with the task's **detectable difference**: on dev the headline resolves 0.023 R², the test
  split only 0.048. Gains of the size we care about (about +0.03 over volumetrics) therefore need confirming on
  another cohort (ADNI), not just on the test split.
- R² is the share of variation explained; AUROC is the chance a progressor scores above a non-progressor (0.5 = coin
  flip); d is a difference in standard deviations. The C1 number converts to a trial size: participants per arm
  needed to detect a 25% slowing of the Alzheimer's-specific change.

## Current results

Run 2026-10-02, dev split, three walnut checkpoints against volumetrics (the eval page has every task, interval and
test score):

| Representation | P1 gain (ΔR²) | P1 scan alone (R²) | p-tau217 (R²) | Tau PET (R²) | Age (R²) | C1 change (d) |
|---|---|---|---|---|---|---|
| Clinical baseline alone | R² 0.274 | | | | | |
| SynthSeg volumes (reference) | +0.046 | 0.153 | 0.066 | 0.034 | 0.36 | 0.47 |
| NeuroQuant volumes | +0.046 | 0.158 | 0.087 | 0.107 | 0.61 | |
| walnut v0.1 ViT-L, medial temporal | +0.047 | 0.170 | 0.139 | 0.074 | 0.50 | 0.28 |
| walnut v0.1 ViT-L, 8 regions | +0.040 | 0.171 | 0.172 | 0.104 | 0.54 | 0.64 |
| walnut v0.1 ViT-L, whole brain | +0.012 | 0.117 | 0.152 | 0.061 | 0.53 | 0.50 |
| walnut v0.1 ViT-B, medial temporal | +0.046 | 0.175 | 0.136 | 0.058 | 0.49 | 0.61 |
| Chance (random representations) | +0.000 | -0.005 | -0.004 | -0.016 | -0.002 | 0.01 |

In words: the foundation models tie volumetrics on the headline. They read more biology from the scan (p-tau217,
amyloid, the hippocampus), but that information overlaps with the blood test already in the clinical baseline, so it
does not add prediction yet. Averaging tokens over the whole brain loses the signal; pooling over the medial temporal
lobe or over regions keeps it. No scan adds to the clinical baseline for CDR progression (AUROC 0.79 with or without).

## Running it

Everything runs on the Sophont cluster with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # Python 3.13 environment from pyproject.toml / uv.lock
uv run python data.py   # step 1: participant tables (10 s)
```

Steps 2-4 run on SLURM (`--qos=high --account=sophont`); the exact commands are at the top of each script.

| Step | Script | Runs | Time |
|---|---|---|---|
| 1 | `data.py` | anywhere | 10 s |
| 2 | `prepare.py`: segment and register every scan (once; already done) | 8 GPUs | 2.5-4 h |
| 3 | `embed.py`: embed every scan with each model in `MODELS` | 8 GPUs | 4 min |
| 4 | `eval.py`: score every representation, write the page | 64 CPUs | 6 min |

`embed.py` and `eval.py` reproduce their outputs exactly. `prepare.py` reproduces the cache closely but not bit for
bit: ANTs registration is multithreaded, so its sums change in the last digits between runs (re-preparing three scans
gave image correlations above 0.9995 with the cache).

**To evaluate a new checkpoint:** add it to `MODELS` in `embed.py` and run step 3, add the same name to `MODELS` in
`eval.py` and run step 4, then open `/data/paul/a4/eval/report.html`. Its three poolings appear as new rows. The
walnut model code comes from a clone of [MedARC-AI/smri-fm](https://github.com/MedARC-AI/smri-fm) at commit 11e53ab
(`/data/paul/a4/smri-fm`, set in `pyproject.toml`); point that at a newer checkout if a model needs newer code.

## Where things are

| What | Where |
|---|---|
| ATRI data release (clinical CSVs, data dictionaries) | `/data/leema/a4/Clinical` |
| A4 T1w scans (BIDS) | `/data/leema/a4/A4-bids/A4` |
| LEARN T1w scans (copied from the team's R2 bucket) | `/data/paul/a4/learn_t1` |
| Prepared scans, SynthSeg volumes and QC, embeddings | `/data/paul/a4/mri/{prepared,synthseg,embed}` |
| Eval tables, frozen split, predictions, results, page | `/data/paul/a4/eval` |
| walnut checkpoints (hf://medarc/walnut) | `/data/smri-datasets/huggingface` |

## Files

| File | Role |
|---|---|
| `data.py` | Builds `participants.parquet` (one row per participant) and `pacc.parquet` (one row per PACC assessment) from the data release |
| `prepare.py` | SynthSeg segmentation and affine registration to the MNI template for all 5,745 scans, cached so new models only need a forward pass |
| `embed.py` | Embeds the prepared scans with each walnut checkpoint and pools the tokens three ways (whole brain, medial temporal lobe, 8 regions) |
| `eval.py` | The eval: frozen split, probes, scores, paired comparisons, chance calibration and resolution; writes `results.json` and the page |
| `report.html` | Template for the eval page (`eval.py` fills it with `results.json`) |

## Data use

A4/LEARN data are under a data use agreement that forbids redistribution. Participant-level files (the parquet
tables, `splits.csv`, `predictions.parquet`, scans) stay under `/data` and must never be committed; `.gitignore`
blocks data file types. `results.json` and the eval page contain aggregate statistics only.
