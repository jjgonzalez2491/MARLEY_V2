import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.family'] = 'Nimbus Sans'

data_path = '/work/cmcc/jg24923/CVaR_58'

checkpoint_runs = [
    "28_CM_CfD_28", "20_CM_CfD_20", "29_CM_CfD_29", # Hybrid+: ND, UD, D
    "11_CM_CfD_11", "19_CM_CfD_19", "12_CM_CfD_12", # Hybrid: ND, UD, D,
    "30_CM_CfD_30", "36_CM_CfD_36", "31_CM_CfD_31", # Hybrid-: ND, UD, D,
    "62_CM_CfD_62", "64_CM_CfD_64", "63_CM_CfD_63", # CRM:   ND, UD, D, 
]

market_designs = {
    "H+":  0,
    "H":   3,
    "H-":  6,
    "CRM": 9,
}

scenarios = {
    "ND":  0,
    "UD":  1,
    "D":   2,
}

scenario_colors = {
    "ND":  "#00A2E9",
    "UD":  "black",
    "D":   "#156082",
}

scenario_ls = {
    "ND":  "-",
    "UD":  "--",
    "D":   "-",
}

# All 5 metrics in display order. The block split (rows 0-1 vs rows 2-4 in this
# list) is realized in the figure via a GridSpec spacer row.
metrics = [
    ("prices.csv",                      "Total Unitary\nCosts",   "[EUR/MWh]", "mean"),
    ("CO2_emissions.csv",               "Cumulative\nEmissions",  "[MtCO$_2$]",   "cumsum"),
    ("res_share.csv",                   "RES\nShare",             "[%]",       "mean"),
    ("storage_share.csv",               "Storage\nShare",         "[%]",       "mean"),
    ("fossil_fuel_capacity_factor.csv", "Fossil Capacity\nFactor","[%]",       "mean"),
]

# Index of the last metric in block 1 (0-indexed). Rows 0..BLOCK1_END belong to
# the upper block (costs + emissions); rows after belong to the lower block.
BLOCK1_END = 1

n_years          = 16
periods_per_year = 6
TIME_SLICE       = slice(6, 6 + n_years * periods_per_year)
x_axis           = np.arange(2025, 2025 + n_years)
x_limit          = 2040


def load_yearly(arr, aggregation):
    n_y = arr.shape[0] // periods_per_year
    reshaped = arr[:n_y * periods_per_year].reshape(n_y, periods_per_year, -1)
    if aggregation == 'mean':
        return reshaped.mean(axis=(1, 2))
    elif aggregation == 'cumsum':
        yearly_per_agent = reshaped.sum(axis=1) / 1e6
        return np.cumsum(yearly_per_agent, axis=0).mean(axis=1)


print("Loading data...")

n_runs    = len(checkpoint_runs)
n_metrics = len(metrics)

data_series = [[None] * n_metrics for _ in range(n_runs)]

for i, folder_name in enumerate(checkpoint_runs):
    folder = os.path.join(data_path, folder_name)
    for j, (csv_file, _, _, aggregation) in enumerate(metrics):
        raw = pd.read_csv(os.path.join(folder, csv_file)).iloc[:, 1:]
        arr = np.array(raw)[TIME_SLICE, :]
        series = load_yearly(arr, aggregation)
        if aggregation == 'mean' and csv_file != 'prices.csv':
            series = series * 100
        data_series[i][j] = series

print("Data loaded.")

# Per-metric y-limits (shared across the 4 market-design columns).
row_ylims = []
for j in range(n_metrics):
    all_vals = np.concatenate([
        data_series[base_idx + scen_offset][j]
        for base_idx in market_designs.values()
        for scen_offset in scenarios.values()
    ])
    lo, hi = np.min(all_vals), np.max(all_vals)
    pad = (hi - lo) * 0.12 if hi != lo else 0.5
    row_ylims.append((lo - pad, hi + pad))

# ── Figure layout ────────────────────────────────────────────────────────────
# 6 grid rows = 2 (block 1) + 1 spacer + 3 (block 2). The spacer is a bit taller
# than half a panel so the row-1 x-tick labels have room to breathe before
# block 2 starts.
n_cols       = len(market_designs)
market_names = list(market_designs.keys())

height_ratios = [2, 2, 0.25, 1, 1, 1]
fig = plt.figure(figsize=(10, 9))
gs  = GridSpec(
    nrows=6, ncols=n_cols,
    figure=fig,
    height_ratios=height_ratios,
    wspace=0.15, hspace=0.15,
)

# Map metric index -> grid row (skipping the spacer at grid row 2).
metric_to_grid_row = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5}
last_row_per_block = {0: 1, 1: 4}  # last metric index in each block
metric_to_block    = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}

axes = {}
for ri, (_, metric_label, ylabel_unit, _) in enumerate(metrics):
    grid_row = metric_to_grid_row[ri]
    for ci, md_name in enumerate(market_names):
        ax = fig.add_subplot(gs[grid_row, ci])
        axes[(ri, ci)] = ax
        base_idx = market_designs[md_name]

        for scen_name, scen_offset in scenarios.items():
            series = data_series[base_idx + scen_offset][ri]
            ax.plot(
                x_axis, series,
                color=scenario_colors[scen_name],
                lw=1.8,
                linestyle=scenario_ls[scen_name],
                zorder=3 if scen_name != "ND" else 2,
                label=scen_name,
            )

        ax.set_xlim(2025, x_limit)
        ax.set_ylim(*row_ylims[ri])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(labelsize=8)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.axvline(x=2028, color='gray', linestyle=':', linewidth=1.5, zorder=4)

        # Column titles only on the very top row of the figure.
        if ri == 0 or ri == 2:
            ax.set_title(md_name, fontsize=10, pad=5, fontweight='bold')

        # Y-label only on column 0; consistent offset across all rows.
        if ci == 0:
            ax.set_ylabel(
                f"{metric_label}\n{ylabel_unit}",
                fontsize=9,
            )
            ax.yaxis.set_label_coords(-0.22, 0.5)
        else:
            ax.yaxis.set_ticklabels([])

        # X-tick labels on the last row of each block (emissions + fossil CF);
        # hidden on all other rows.
        is_block_bottom = ri in last_row_per_block.values()
        if not is_block_bottom:
            ax.xaxis.set_ticklabels([])

# ── Legend & x-axis label ────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=scenario_colors[scen], lw=2.0,
           linestyle=scenario_ls[scen], label=scen)
    for scen in scenarios
]

# Add the decree deactivation line
legend_handles.append(
    Line2D([0], [0], color='gray', lw=1.5, linestyle=':', label='Restoration of the Carbon Price Signal\n[UD] scenario')
)

fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=5,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02),
)
fig.supxlabel('Year', fontsize=10, y=0.06)

plt.savefig(
    "plot_bars_variable_evolutions_V5.pdf",
    format="pdf", bbox_inches="tight"
)
plt.show()
print("Saved: plot_bars_variable_evolutions_V5.pdf")