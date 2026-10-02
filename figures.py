"""Export the README's dev comparisons from aggregate results: uv run --locked --group docs python figures.py."""
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

RESULTS = Path("/data/paul/a4/eval/results.json")
OUT = Path(__file__).with_name("readme-results.svg")

raw = RESULTS.read_bytes()
r = json.loads(raw)
reference = r["reference"]
reps = list(r["reps"])
prognosis = {x["rep"]: x for x in r["results"] if x["task"] == "P1" and x["mode"] == "base"}
change = {x["rep"]: x["dev"] for x in r["change"]}
assert reference in reps and all(rep in prognosis for rep in reps)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
                     "svg.hashsalt": "a4-learn-readme", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.edgecolor": "#ccd5dd", "text.color": "#172b3a",
                     "axes.labelcolor": "#172b3a", "xtick.color": "#526575", "ytick.color": "#172b3a"})
fig, axes = plt.subplots(1, 3, figsize=(14, 7.4), sharey=True)
fig.subplots_adjust(left=0.29, right=0.98, bottom=0.24, top=0.78, wspace=0.25)
fig.text(0.02, 0.96, "What does MRI add beyond conventional volumes?", fontsize=19, weight="bold")
fig.text(0.02, 0.92, f"A4/LEARN · {r['config']['run_date']} · dev cross-validation · all included representations", color="#526575")

for ax, key, title, unit, ref_value, spacing in zip(
    axes, ["P1 base", "C d", "C rho_decline"],
    ["P1 · Predict decline", "C1 · Detect cohort change", "C2 · Track decline"],
    ["Difference in R²", "Difference in d", "Difference in Spearman ρ"],
    [f"SynthSeg gain: +{prognosis[reference]['gain']['dev']:.3f} R²",
     f"SynthSeg separation: d = {change[reference]['d']:.2f}",
     f"SynthSeg correlation: ρ = {change[reference]['rho_decline']:.2f}"],
    [0.02, 0.2, 0.2],
):
    ax.set_title(title + "\n" + ref_value, loc="left", fontsize=11, pad=17, linespacing=1.7)
    ax.axvline(0, color="#526575", linewidth=1.1, zorder=1)
    ax.xaxis.set_major_locator(MultipleLocator(spacing))
    ax.grid(axis="x", color="#e5ebf0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(unit + "\n← worse     vs SynthSeg     better →", fontsize=10, labelpad=10)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(reps) - 0.4, -0.6)
    for i, rep in enumerate(reps):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f7f9", zorder=0)
        if key != "P1 base" and rep not in change:
            ax.text(0.5, i, "baseline only", transform=ax.get_yaxis_transform(), ha="center", color="#788997", fontsize=10)
            continue
        if rep == reference:
            ax.plot(0, i, "D", color="#172b3a", markersize=6, zorder=3)
            continue
        if key == "P1 base":
            pair = prognosis[rep]["vs_reference"]
            difference, se = pair["dev"], pair["dev_se"]
        else:
            pair = change[rep]["vs_reference"][key.removeprefix("C ")]
            difference, se = pair["diff"], pair["se"]
        # Match report.html's widen(): bootstrap SE plus the variance of two random-probe fits.
        interval = 1.96 * np.sqrt(se ** 2 + 2 * r["meta"]["null"][key]["sd"] ** 2)
        assert np.isfinite([difference, interval]).all()
        ax.errorbar(difference, i, xerr=interval, fmt="o", color="#17678a" if rep.startswith("walnut") else "#637281",
                    markersize=5, linewidth=1.6, capsize=3, zorder=3)
    ax.margins(x=0.1)

axes[0].set_yticks(range(len(reps)), [rep.replace(", ", " / ") for rep in reps], fontsize=10)
fig.text(0.02, 0.075, "Points: paired differences from SynthSeg. Bars: approximate 95% intervals, including random-probe variation.", fontsize=10)
fig.text(0.02, 0.044, "P1 uses baseline MRI; C1–C2 use week-240 change. A4–LEARN separation does not establish treatment benefit.", fontsize=10, color="#526575")
fig.savefig(OUT, facecolor="white", metadata={"Date": None, "Title": "A4/LEARN dev comparisons with SynthSeg volumes",
            "Description": f"Aggregate results SHA256 {hashlib.sha256(raw).hexdigest()}; split SHA256 {r['meta']['split_sha']}. "
                           "P1: gain over clinical baseline. C1: age-adjusted A4–LEARN change separation. C2: correlation with decline."})
OUT.write_text("\n".join(line.rstrip() for line in OUT.read_text().splitlines()) + "\n")
print(f"Wrote {OUT} from {RESULTS} ({r['config']['run_date']})")
