# Step 1 of 4. Builds the two participant-level tables the eval needs from the A4/LEARN data release.
#
# The study. A4 ("Anti-Amyloid Treatment in Asymptomatic Alzheimer's") screened about 4,500 cognitively unimpaired
# people aged 65-85 with amyloid PET. The 1,169 with elevated brain amyloid (early, preclinical Alzheimer's) were
# randomized to the anti-amyloid antibody solanezumab or placebo for 240 weeks; the drug did not slow decline.
# LEARN followed 538 people who screened amyloid-negative, untreated, with the same tests and MRI schedule, as a
# reference for normal ageing. Visits are numbered: 1-5 screening, 6 = randomization (week 0), then about every
# 12 weeks up to 66 = week 240, the end of the placebo-controlled ("blinded") phase; an open-label phase followed.
# LEARN uses the same numbers for its matched visits.
#
# Input: the ATRI release A4LEARN 1.2.20260114 (CSV files in "Derived Data", "Raw Data" and "External Data"; variable
# definitions in "Documents/Data Dictionaries"). Output, under a data use agreement (never commit or share these):
#   participants.parquet  one row per A4 or LEARN participant, indexed by BID (the study's participant ID)
#   pacc.parquet          one row per participant x PACC assessment, for the decline-rate target
import numpy as np
import pandas as pd

RAW = "/data/leema/a4/Clinical/"
OUT = "/data/paul/a4/eval/"
COHORTS = ["A4", "LEARN"]  # the release also covers screen failures ("SF"), which the eval does not use

# ---------------- who each participant is ----------------
info = pd.read_csv(RAW + "Derived Data/SUBJINFO.csv").query("SUBSTUDY in @COHORTS").set_index("BID")
P = pd.DataFrame(index=info.index)
P["cohort"] = info.SUBSTUDY                                 # A4 (amyloid-positive) or LEARN (amyloid-negative)
P["arm"] = np.where(info.SUBSTUDY == "A4", info.TX, "LEARN")  # Placebo, Solanezumab or LEARN
P["treated"] = (info.TX == "Solanezumab").astype(float)
P["mitt"] = info.MITTFL == 1           # modified intention-to-treat: randomized, with a baseline and a later assessment
P["tau_substudy"] = info.TAUPETFL == 1  # A4 participants who also had tau PET
P["age"] = info.AGEYR
P["female"] = (info.SEX == 1).astype(float)
P["edu"] = info.EDCCNTU                 # years of education
P["apoe4"] = info.APOEGNPRSNFLG         # carries the APOE e4 risk allele (1/0; missing if not genotyped)
P["centiloid"] = info.AMYLCENT          # amyloid load on the screening PET, in centiloids (0 = typical amyloid-negative)

# ---------------- cognition and everyday function ----------------
# ADQS is ATRI's analysis dataset of every questionnaire and test score (one row per participant x test x visit).
# PACC, the trial's primary outcome, is the Preclinical Alzheimer Cognitive Composite: the mean of four z-scored tests
# (word-list recall, story recall, digit-symbol coding, MMSE), so 0 is the average baseline score and lower is worse.
# Alternating test forms (A, B and SC) are used to limit practice effects; eval.py corrects for them.
q = pd.read_csv(RAW + "Derived Data/ADQS.csv", low_memory=False).query("SUBSTUDY in @COHORTS")
q["QSTESTCD"] = q.QSTESTCD.replace({"CDSOB": "CDRSB"})  # LEARN codes the CDR sum of boxes as CDSOB
scores = q.dropna(subset=["QSSTRESN"]).drop_duplicates(["BID", "QSTESTCD", "AVISIT"])
scores = scores.pivot(index="BID", columns=["QSTESTCD", "AVISIT"], values="QSSTRESN")
P["pacc_bl"] = scores[("PACC", 6.0)]       # PACC at randomization
P["cfisp_bl"] = scores[("CFISP", 1.0)]     # Cognitive Function Index (cognitive complaints), study partner's report
P["cfipt_bl"] = scores[("CFIPT", 1.0)]     # ... participant's own report
P["cdrsb_bl"] = scores[("CDRSB", 1.0)]     # Clinical Dementia Rating, sum of boxes (0 = no impairment)
P["adlpqsp_bl"] = scores[("ADLPQSP", 1.0)]  # ADCS activities of daily living (prevention version), study partner's report

# the four PACC tests themselves, at randomization
tests = pd.read_csv(RAW + "Derived Data/PACC.csv").query("SUBSTUDY in @COHORTS and VISCODE == 6").set_index("BID")
P["fcsrt96_bl"] = tests.FCTOTAL96       # Free and Cued Selective Reminding Test, total recall (0-96)
P["lm_delayed_bl"] = tests.LDELTOTAL    # Logical Memory delayed story recall
P["dsst_bl"] = tests.DIGITTOTAL         # Digit Symbol Substitution Test
P["mmse_v6_bl"] = tests.MMSCORE         # Mini-Mental State Examination (0-30)

# Cogstate C3: a computerized composite of memory and attention tasks, averaged over the two screening sessions
cog = pd.read_csv(RAW + "External Data/cogstate_battery.csv", low_memory=False, usecols=["SUBSTUDY", "BID", "VISCODE", "C3Comp"])
cog = cog.query("SUBSTUDY in @COHORTS").dropna(subset=["C3Comp"]).groupby(["BID", "VISCODE"], as_index=False).first()
P["c3_bl"] = cog[cog.VISCODE.isin([1, 3])].groupby("BID").C3Comp.mean()

# ---------------- outcomes ----------------
# PACC change from randomization to the scheduled week-240 visit, inside the blinded phase (completers only)
week240 = q.query("QSTESTCD == 'PACC' and AVISIT == 66 and EPOCH in ['BLINDED TREATMENT', 'Analysis visit <= 66']")
week240 = week240.dropna(subset=["QSSTRESN"]).set_index("BID")
P["pacc_chg240"] = week240.QSSTRESN.reindex(P.index) - P.pacc_bl

# Progression on the Clinical Dementia Rating (CDR global: 0 = normal, 0.5 = very mild impairment, 1+ = dementia).
# ATRI's CDEVENT = 1 when the global score is above 0 at two consecutive blinded visits (or at the last one);
# CDADTC_DAYS_T0 is the day of that event, or of the last assessment for people without one.
cdr = pd.read_csv(RAW + "Raw Data/cdr.csv", low_memory=False).query("SUBSTUDY in @COHORTS")
cdr = cdr.dropna(subset=["CDEVENT"]).set_index("BID")
assert cdr.index.is_unique
P["cdr_event"] = cdr.CDEVENT
P["cdr_weeks"] = cdr.CDADTC_DAYS_T0 / 7

# ---------------- blood and PET biomarkers at baseline ----------------
# Plasma p-tau217 (Lilly assay, U/mL), the strongest blood marker of Alzheimer's pathology; baseline = A4 visit 6,
# LEARN visit 1. ORRESRAW keeps the measured value even when the lab flagged it as outside its calibrated range.
ptau = pd.read_csv(RAW + "External Data/biomarker_pTau217.csv", low_memory=False).query("SUBSTUDY in @COHORTS")
ptau = ptau[ptau.STAT.isna()]  # STAT marks samples that were not analysed
ptau = ptau[((ptau.SUBSTUDY == "A4") & (ptau.VISCODE == 6)) | ((ptau.SUBSTUDY == "LEARN") & (ptau.VISCODE == 1))].set_index("BID")
assert ptau.index.is_unique
P["ptau217_bl"] = pd.to_numeric(ptau.ORRESRAW)

# Tau PET (flortaucipir) at the first tau scan (A4 visit 4, LEARN visit 6), two summary measures:
# the MUBADA composite from Avid (cerebellar-crus reference region) ...
tau = pd.read_csv(RAW + "External Data/imaging_SUVR_tau.csv", low_memory=False)
tau = tau.query("SUBSTUDY in @COHORTS and brain_region == 'MUBADA Mask'")
tau = tau[((tau.SUBSTUDY == "A4") & (tau.VISCODE == 4)) | ((tau.SUBSTUDY == "LEARN") & (tau.VISCODE == 6))].set_index("BID")
assert tau.index.is_unique
P["tau_mubada_bl"] = tau.suvr_crus
# ... and the temporal meta-ROI from Stanford's FreeSurfer pipeline (Jack et al.): volume-weighted mean SUVR over the
# amygdala and the entorhinal, parahippocampal, fusiform, inferior- and middle-temporal cortex
stanford = pd.read_csv(RAW + "External Data/imaging_Tau_PET_Stanford.csv").query("SUBSTUDY in @COHORTS").set_index("BID")
rois = ["Left.Amygdala", "Right.Amygdala"] + [f"ctx.{h}.{r}" for h in ["lh", "rh"]
                                              for r in ["entorhinal", "parahippocampal", "fusiform", "inferiortemporal", "middletemporal"]]
volume = stanford[["Volume_mm3." + r for r in rois]].to_numpy()
P["tau_metatemporal_bl"] = pd.Series((stanford[["Mean." + r for r in rois]].to_numpy() * volume).sum(1) / volume.sum(1), index=stanford.index)
assert "B34660963" in P.index  # (.loc would otherwise silently add a row)
P.loc["B34660963", ["tau_mubada_bl", "tau_metatemporal_bl"]] = np.nan  # off-target binding outside the brain (Stanford methods appendix)

# ---------------- NeuroQuant MRI volumes at baseline ----------------
# NeuroQuant is an FDA-cleared volumetry product; the release has its regional volumes (cm^3) for the screening MRI
# (A4 visit 4; LEARN's single record has no visit code). -4 codes a missing value; zero is only valid for
# white-matter hypointensities. HOC is the hippocampal occupancy ratio.
nq = pd.read_csv(RAW + "External Data/imaging_volumetric_mri.csv").query("SUBSTUDY in @COHORTS")
regions = nq.columns.drop(["SUBSTUDY", "BID", "VISCODE", "Date_DAYS_CONSENT", "Date_DAYS_T0"])
nq[regions] = nq[regions].where((nq[regions] > 0) | ((nq[regions] == 0) & regions.str.contains("WMHypo")))
nq = nq[((nq.SUBSTUDY == "A4") & (nq.VISCODE == 4)) | ((nq.SUBSTUDY == "LEARN") & nq.VISCODE.isna())].set_index("BID")
assert nq.index.is_unique
P = P.join(nq[regions].add_prefix("nq_"))

# ---------------- PACC at every assessment, for the decline-rate target ----------------
phase = {"SCREENING": "screening", "BLINDED TREATMENT": "blinded", "OPEN LABEL TREATMENT": "open-label",
         "Analysis visit <= 66": "blinded", "Analysis visit > 66": "open-label"}  # LEARN names its epochs by visit
pacc = q[(q.QSTESTCD == "PACC") & q.BID.isin(P.index)].dropna(subset=["QSSTRESN"])
pacc = pd.DataFrame({"BID": pacc.BID, "visit": pacc.AVISIT, "week": pacc.ADURW, "pacc": pacc.QSSTRESN,
                     "form": pacc.QSVERSION, "phase": pacc.EPOCH.map(phase)})

P.to_parquet(OUT + "participants.parquet")
pacc.to_parquet(OUT + "pacc.parquet", index=False)
print(P.groupby("arm").size().to_string(), f"\n{len(pacc):,} PACC assessments", flush=True)
