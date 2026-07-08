"""
nd_vs_d_table_and_significance.py
=============================================================================
ONE script, TWO perfectly-aligned outputs for the [ND] vs [D] comparison:

  (1) MAIN-TEXT TABLE  -- reproduces
      plot_metrics_final_cost_markets_lobby_unified_V20_table.py:
      per-scenario  mean [p25, p75]  across simulations, and the percentage
      difference (D - ND)/ND, grouped by market, for each window.

  (2) APPENDIX SIGNIFICANCE TABLE -- the statistical backing for (1):
      Welch + Mann-Whitney tests, Holm correction across markets, Hedges' g
      and Cliff's delta with magnitude bands, a 95% bootstrap CI on the
      percentage difference, and a verdict.

ALIGNMENT GUARANTEE
-------------------
Both outputs are built from the SAME per-simulation vectors, with the SAME
metric definitions as the main figure / V20 table:
    COST      = mean over the window of prices.csv[step, sim]          [EUR/MWh]
                (prices.csv already includes carbon-tax returns and all
                 mechanism settlements -- it is the total system cost)
    EMISSIONS = sum over the window of CO2_emissions.csv / (n_years*1e6) [MtCO2/yr]
Consequently the main table's `Diff_pct` and the appendix's `pct_DminusND`
are the SAME number for every cell (a sanity check at the end asserts this).

DESIGN (drives the statistics)
------------------------------
Each scenario is an independent training session; simulations are NOT seed-
aligned across scenarios, so the comparison is UNPAIRED. The sample per
scenario is the ~1000 simulations (columns of the CSVs).

A NOTE ON LARGE n
-----------------
With ~1000 simulations the standard error is tiny and almost any non-zero
difference is "significant"; the substantive results are the magnitude
(percentage difference + CI) and the effect size, not the p-value. The verdict
separates "significant AND non-negligible" from "significant but negligible".

Sign convention everywhere: (D - ND), i.e. the effect of the Decreto, reported
relative to the [ND] baseline -- matching the main-text table.

Outputs (no LaTeX; build the tables yourself from these):
  * nd_vs_d_table_cost.csv          (Market, Window, ND, D, Diff_pct)
  * nd_vs_d_table_emissions.csv     (Market, Window, ND, D, Diff_pct)
  * nd_vs_d_significance_appendix.csv  (long, 32 rows, slim columns)
  plus pretty-prints of all three to stdout.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# CONFIG
# ============================================================================
data_path = os.environ.get("MARLEY_STATS_DATA_PATH", "/work/cmcc/jg24923/CVaR_58/")

# market -> (ND run folder, D run folder). Order is the main-table order.
MARKETS = [
    ("H+",  "28_CM_CfD_28", "29_CM_CfD_29"),
    ("H",   "11_CM_CfD_11", "12_CM_CfD_12"),
    ("H-",  "30_CM_CfD_30", "31_CM_CfD_31"),
    ("CRM", "62_CM_CfD_62", "63_CM_CfD_63"),
]
length_sim = 102
WINDOWS = [('Aggregate', 6, 102), ('Short', 6, 18),
           ('Mid', 18, 66), ('Long', 66, 102)]

# ---- statistics config ----
ALPHA       = 0.05
N_BOOT      = 10_000
RNG_SEED    = 12345
MC_METHOD   = "holm"
G_BANDS     = [(0.2, "negligible"), (0.5, "small"), (0.8, "medium")]      # else "large"
NONNEGLIGIBLE_G = 0.2

OUT_COST   = "standard_difference_2_cost.csv"
OUT_EMISS  = "standard_difference_2_emissions.csv"
OUT_APPDX  = "standard_difference_appendix.csv"

rng = np.random.default_rng(RNG_SEED)

# ============================================================================
# DATA LOADING  (cost from prices.csv, emissions from CO2_emissions.csv)
# ============================================================================
price_raw = {}   # run-folder -> (steps, n_sims)
emiss_raw = {}   # run-folder -> (steps, n_sims)

all_runs = [r for _, nd, d in MARKETS for r in (nd, d)]
print(f"Loading data from: {data_path}\n")
for run in all_runs:
    folder = os.path.join(data_path, run)
    p = np.array(pd.read_csv(os.path.join(folder, 'prices.csv')).iloc[:, 1:])
    e = np.array(pd.read_csv(os.path.join(folder, 'CO2_emissions.csv')).iloc[:, 1:])
    price_raw[run] = p[:length_sim, :]
    emiss_raw[run] = e[:length_sim, :]
    print(f"  {run:<16}  price sims={price_raw[run].shape[1]:>5}  "
          f"emiss sims={emiss_raw[run].shape[1]:>5}  steps={price_raw[run].shape[0]}")
print()

# ============================================================================
# PER-SIMULATION METRICS  (identical definitions feed BOTH tables)
# ============================================================================
def per_sim_cost(run, s, e):
    """Mean over the window of prices.csv  ->  EUR/MWh per simulation."""
    return np.mean(price_raw[run][s:e, :], axis=0)

def per_sim_emiss(run, s, e):
    """Annualised emissions  ->  MtCO2/year per simulation."""
    n_years = (e - s) / 6.0
    return np.sum(emiss_raw[run][s:e, :], axis=0) / (n_years * 1e6)

METRICS = {
    "cost":      dict(fn=per_sim_cost,  label="Total system cost [EUR/MWh]"),
    "emissions": dict(fn=per_sim_emiss, label="Yearly emissions [MtCO2/yr]"),
}

# ============================================================================
# EFFECT SIZES (computed in the (D - ND) direction)
# ============================================================================
def hedges_g(d, nd):
    n1, n2 = len(d), len(nd)
    s1, s2 = np.var(d, ddof=1), np.var(nd, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if pooled <= 0:
        return np.nan
    g = (np.mean(d) - np.mean(nd)) / np.sqrt(pooled)
    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return J * g

def cliffs_delta(d, nd):
    """P(D>ND) - P(D<ND), via the Mann-Whitney U relation (no NxM matrix)."""
    n1, n2 = len(d), len(nd)
    pooled = np.concatenate([d, nd])
    ranks = stats.rankdata(pooled)
    U1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return 2.0 * U1 / (n1 * n2) - 1.0

def band(value, bands):
    v = abs(value)
    for thr, name in bands:
        if v < thr:
            return name
    return "large"

# ============================================================================
# BOOTSTRAP CI ON THE PERCENTAGE DIFFERENCE  (D - ND)/ND * 100
# ============================================================================
def boot_ci_pct(d, nd, nboot, alpha):
    n1, n2 = len(d), len(nd)
    i1 = rng.integers(0, n1, size=(nboot, n1))
    i2 = rng.integers(0, n2, size=(nboot, n2))
    md = d[i1].mean(axis=1)
    mn = nd[i2].mean(axis=1)
    pct = (md - mn) / mn * 100.0
    lo, hi = np.percentile(pct, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi

# ============================================================================
# DESCRIPTIVE + SIGNIFICANCE PER CELL
# ============================================================================
def descriptive(v):
    return dict(mean=float(np.mean(v)),
                p25=float(np.percentile(v, 25)),
                p75=float(np.percentile(v, 75)))

def significance(d, nd):
    d  = np.asarray(d,  float)
    nd = np.asarray(nd, float)
    out = {}
    out["p_welch"] = stats.ttest_ind(d, nd, equal_var=False).pvalue
    out["p_mwu"]   = stats.mannwhitneyu(d, nd, alternative="two-sided").pvalue
    g = hedges_g(d, nd)
    out["hedges_g"]  = g
    out["g_band"]    = band(g, G_BANDS)
    out["cliffs_delta"] = cliffs_delta(d, nd)
    out["ci_lo"], out["ci_hi"] = boot_ci_pct(d, nd, N_BOOT, ALPHA)
    return out

# ---- multiple-comparison correction ----
def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.0)
    return adj

def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]; run = min(run, p[idx] * m / (rank + 1)); adj[idx] = min(run, 1.0)
    return adj

def verdict(p_corr, g):
    if not np.isfinite(p_corr):
        return "undefined"
    sig = p_corr < ALPHA
    nonneg = abs(g) >= NONNEGLIGIBLE_G
    if sig and nonneg:  return "significant"
    if sig:             return "sig. but negligible"
    return "n.s."

# ============================================================================
# BUILD EVERYTHING (single pass over the shared per-sim vectors)
# ============================================================================
cell = lambda s: f"{s['mean']:.2f} [{s['p25']:.2f}, {s['p75']:.2f}]"

desc_rows = []   # for the main-text tables
app_rows  = []   # for the appendix table

for metric_key, meta in METRICS.items():
    fn = meta["fn"]
    for win_name, s, e in WINDOWS:
        family = []   # the 4 markets -> for Holm correction within this (metric, window)
        for mkt, nd_run, d_run in MARKETS:
            nd_v = fn(nd_run, s, e)
            d_v  = fn(d_run,  s, e)

            nd_s, d_s = descriptive(nd_v), descriptive(d_v)
            pct = (d_s["mean"] - nd_s["mean"]) / nd_s["mean"] * 100.0 \
                  if nd_s["mean"] != 0 else np.nan

            desc_rows.append(dict(Metric=meta["label"], Market=mkt, Window=win_name,
                                  ND=cell(nd_s), D=cell(d_s), Diff_pct=round(pct, 2)))

            sig = significance(d_v, nd_v)
            sig.update(metric=metric_key, market=mkt, window=win_name,
                       pct=pct, mean_nd=nd_s["mean"], mean_d=d_s["mean"],
                       sd_nd=float(np.std(nd_v, ddof=1)),
                       sd_d=float(np.std(d_v, ddof=1)),
                       n_nd=len(nd_v), n_d=len(d_v))
            family.append(sig)

        # Holm + BH across the 4 markets in this (metric, window)
        for ptype in ("p_welch", "p_mwu"):
            praw = [r[ptype] for r in family]
            for r, a, b in zip(family, holm(praw), benjamini_hochberg(praw)):
                r[f"{ptype}_holm"] = a
                r[f"{ptype}_bh"]   = b
        for r in family:
            r["verdict"] = verdict(r[f"p_mwu_{MC_METHOD}"], r["hedges_g"])
        app_rows.extend(family)

desc_df = pd.DataFrame(desc_rows)
app_df  = pd.DataFrame(app_rows)

# ============================================================================
# OUTPUT 1: MAIN-TEXT TABLES (one CSV per metric, V20 format)
# ============================================================================
for metric_key, meta in METRICS.items():
    sub = (desc_df[desc_df.Metric == meta["label"]]
           .drop(columns=["Metric"]).reset_index(drop=True))
    fname = OUT_COST if metric_key == "cost" else OUT_EMISS
    sub.to_csv(fname, index=False)
    print(f"Saved main-text table: {fname}")

print("\n" + "=" * 96)
print("MAIN-TEXT TABLE  (per-scenario  mean [p25, p75];  Diff% = (D - ND)/ND)")
print("=" * 96)
print(f"{'Market':7}{'Metric':30}{'Window':10}"
      f"{'ND  mean [p25,p75]':26}{'D  mean [p25,p75]':26}{'Diff%':>8}")
print("-" * 107)
prev = None
for _, r in desc_df.iterrows():
    mc = r['Market'] if r['Market'] != prev else ''
    prev = r['Market']
    print(f"{mc:7}{r['Metric']:30}{r['Window']:10}"
          f"{r['ND']:26}{r['D']:26}{r['Diff_pct']:>7.2f}%")

# ============================================================================
# OUTPUT 2: APPENDIX SIGNIFICANCE TABLE (slim, long, (D - ND) convention)
# ============================================================================
def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

appendix = pd.DataFrame({
    "window":       app_df["window"],
    "market":       app_df["market"],
    "metric":       app_df["metric"],
    "pct_DminusND": app_df["pct"].round(2),
    "ci95_low":     app_df["ci_lo"].round(2),
    "ci95_high":    app_df["ci_hi"].round(2),
    "p_holm":       app_df[f"p_mwu_{MC_METHOD}"].map(fmt_p),
    "hedges_g":     app_df["hedges_g"].round(2),
    "magnitude":    app_df["g_band"],
    "verdict":      app_df["verdict"],
})
# order: market block, window, cost before emissions (matches main table)
mkt_order = {m: k for k, (m, _, _) in enumerate(MARKETS)}
win_order = {w: k for k, (w, _, _) in enumerate(WINDOWS)}
met_order = {"cost": 0, "emissions": 1}
appendix = appendix.sort_values(
    by=["market", "window", "metric"],
    key=lambda c: c.map({**mkt_order, **win_order, **met_order}).fillna(c)
).reset_index(drop=True)
appendix.to_csv(OUT_APPDX, index=False)
print(f"\nSaved appendix table:  {OUT_APPDX}")

print("\n" + "=" * 104)
print("APPENDIX SIGNIFICANCE TABLE  (UNPAIRED;  (D - ND) vs ND;  "
      f"Mann-Whitney p, {MC_METHOD}-corrected across markets)")
print("=" * 104)
print(f"{'window':11}{'market':7}{'metric':11}{'%Δ(D-ND)':>10}"
      f"{'95% CI':>20}{'p_holm':>9}{'g':>7}{'magnitude':>12}  verdict")
for _, r in appendix.iterrows():
    ci = f"[{r['ci95_low']:+.2f},{r['ci95_high']:+.2f}]"
    print(f"{r['window']:11}{r['market']:7}{r['metric']:11}{r['pct_DminusND']:>+9.2f}%"
          f"{ci:>20}{r['p_holm']:>9}{r['hedges_g']:>+7.2f}"
          f"{r['magnitude']:>12}  {r['verdict']}")

n_per = int(app_df[["n_nd", "n_d"]].to_numpy().min())
print(f"\nSimulations per group (min): {n_per}.  g = Hedges' g (D-ND); magnitude band is for g.")
print("At this sample size read %Δ, its CI, and the magnitude as the result; "
      "p is near-uniformly tiny.")

# ============================================================================
# ALIGNMENT SANITY CHECK: main-table Diff_pct == appendix pct_DminusND
# ============================================================================
mism = 0
label_to_key = {v["label"]: k for k, v in METRICS.items()}
app_lookup = {(r["market"], r["window"], r["metric"]): r["pct_DminusND"]
              for _, r in appendix.iterrows()}
for _, r in desc_df.iterrows():
    key = (r["Market"], r["Window"], label_to_key[r["Metric"]])
    if abs(app_lookup[key] - r["Diff_pct"]) > 1e-9:
        mism += 1
print(f"\nAlignment check: {len(desc_df)} cells, {mism} mismatches between "
      "main-table Diff_pct and appendix pct_DminusND (expected 0).")