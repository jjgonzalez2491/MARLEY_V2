import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec

matplotlib.rcParams['font.family'] = 'Nimbus Sans'

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

data_path = '/work/cmcc/jg24923/CVaR_58/'

number_runs = 8

checkpoint_runs = [
    "28_CM_CfD_28", "29_CM_CfD_29", "11_CM_CfD_11", "12_CM_CfD_12",
    "30_CM_CfD_30", "31_CM_CfD_31", "62_CM_CfD_62", "63_CM_CfD_63",
]

run_order_names = ["ND", "D", "", "ND", "D", "", "ND", "D", "", "ND", "D"]
group_labels = ["H+", "H", "H-", "CRM"]
run_order = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8)

years = [2030, 2035, 2040]
n_years = len(years)
n_tech = 8

# Outer grid layout (matches the original 4x2 ordering).
# Each entry: (tech_idx_in_data, title, outer_row, outer_col).
# Tech indices follow the original CSV column order:
#   0 Solar PV, 1 Onshore Wind, 2 Hydro, 3 Coal,
#   4 OCGT,    5 CCGT,           6 Bat-3h, 7 Bat-8h
panel_spec = [
    (0, "Solar PV",            0, 0),
    (1, "Onshore Wind",        1, 0),
    (2, "Offshore Wind",       2, 0),
    (6, "Batteries - 3 Hours", 3, 0),
    (3, "Coal",                0, 1),
    (4, "OCGT",                1, 1),
    (5, "CCGT",                2, 1),
    (7, "Batteries - 8 Hours", 3, 1),
]

n_outer_rows = 4
n_outer_cols = 2

colors = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5', '#00A2E9',
          '#003F5C', '#00E5FF', '#00B89F', '#008CFF']

# -----------------------------------------------------------------------------
# Bar positions (8 bars across 4 market-design groups, with gaps between groups)
# -----------------------------------------------------------------------------

x_positions_MARL = [0, 1, 3, 4, 6, 7, 9, 10]
separator_positions = [2, 5, 8]
group_positions = [0.5, 3.5, 6.5, 9.5]

# -----------------------------------------------------------------------------
# Data containers: one dict per year, each holding all the stats we need
# -----------------------------------------------------------------------------

def make_year_arrays():
    return {
        "mean":        np.zeros([number_runs, n_tech]),
        "min":         np.zeros([number_runs, n_tech]),
        "max":         np.zeros([number_runs, n_tech]),
        "p25":         np.zeros([number_runs, n_tech]),
        "p75":         np.zeros([number_runs, n_tech]),
        "merchant":    np.zeros([number_runs, n_tech]),
        "cm":          np.zeros([number_runs, n_tech]),
        "CfD":         np.zeros([number_runs, n_tech]),
        "flexibility": np.zeros([number_runs, n_tech]),
        "existing":    np.zeros([number_runs, n_tech]),
        "total":       np.zeros([number_runs, n_tech]),
    }

data = {y: make_year_arrays() for y in years}

# -----------------------------------------------------------------------------
# Load data: same logic as the original 2040 script, applied uniformly per year
# -----------------------------------------------------------------------------

for i, run_folder in enumerate(checkpoint_runs):
    folder = os.path.join(data_path, run_folder)

    for year in years:
        # Total capacity distribution (across simulation seeds/scenarios)
        cap_total = pd.read_csv(os.path.join(folder, f'capacity_{year}.csv'))
        cap_total = np.array(cap_total.iloc[:, 1:]) / 1e3

        d = data[year]
        d["mean"][i, :] = np.mean(cap_total, axis=1)
        # Original 2040 script used 5/95 percentiles for "min"/"max"; preserved.
        d["min"][i, :] = np.percentile(cap_total, 5,  axis=1)
        d["max"][i, :] = np.percentile(cap_total, 95, axis=1)
        d["p25"][i, :] = np.percentile(cap_total, 25, axis=1)
        d["p75"][i, :] = np.percentile(cap_total, 75, axis=1)

        # Source-decomposition components (mean across seeds)
        merchant_tmp    = pd.read_csv(os.path.join(folder, f'capacity_{year}_merchant.csv'))
        cm_tmp          = pd.read_csv(os.path.join(folder, f'capacity_{year}_cm.csv'))
        cfd_tmp         = pd.read_csv(os.path.join(folder, f'capacity_{year}_CfD.csv'))
        flexibility_tmp = pd.read_csv(os.path.join(folder, f'capacity_{year}_flexibility.csv'))

        d["merchant"][i, :]    = np.mean(np.array(merchant_tmp.iloc[:, 1:])    / 1e3, axis=1)
        d["cm"][i, :]          = np.mean(np.array(cm_tmp.iloc[:, 1:])          / 1e3, axis=1)
        d["CfD"][i, :]         = np.mean(np.array(cfd_tmp.iloc[:, 1:])         / 1e3, axis=1)
        d["flexibility"][i, :] = np.mean(np.array(flexibility_tmp.iloc[:, 1:]) / 1e3, axis=1)

        # Existing = Total - (sum of new build by source)
        d["existing"][i, :] = (
            d["mean"][i, :]
            - d["merchant"][i, :]
            - d["cm"][i, :]
            - d["CfD"][i, :]
            - d["flexibility"][i, :]
        )

# Clip negative existing capacities to zero (numerical noise)
for year in years:
    data[year]["existing"] = np.clip(data[year]["existing"], 0, None)
    data[year]["total"]    = data[year]["mean"]

# -----------------------------------------------------------------------------
# Plotting helper: one inner sub-panel = one (technology, year) cell
# -----------------------------------------------------------------------------

def plot_panel(ax, d, tech_idx,
               show_xlabels=False, show_group_labels=False, show_ylabel=False):
    """Render one (technology x year) sub-panel: stacked bars + whiskers."""

    existing    = d["existing"][run_order,    tech_idx]
    merchant    = d["merchant"][run_order,    tech_idx]
    cm          = d["cm"][run_order,          tech_idx]
    cfd         = d["CfD"][run_order,         tech_idx]
    flexibility = d["flexibility"][run_order, tech_idx]
    total       = d["total"][run_order,       tech_idx]

    # Stacked bars (Existing -> Merchant -> CM -> CfD -> Flexibility)
    ax.bar(x_positions_MARL, existing,
           color=colors[0], edgecolor=colors[0], label="Existing")
    ax.bar(x_positions_MARL, merchant,
           bottom=existing,
           color=colors[2], edgecolor=colors[2], label="Merchant")
    ax.bar(x_positions_MARL, cm,
           bottom=existing + merchant,
           color='white', hatch='xx', edgecolor=colors[3], label="CM")
    ax.bar(x_positions_MARL, cfd,
           bottom=existing + merchant + cm,
           color=colors[4], edgecolor=colors[4], label="CfD")
    ax.bar(x_positions_MARL, flexibility,
           bottom=existing + merchant + cm + cfd,
           color='white', hatch='oo', edgecolor=colors[5], label="Flexibility")

    height = existing + merchant + cm + cfd + flexibility

    # Min-max whisker (no caps)
    err_lo = total - d["min"][run_order, tech_idx]
    err_hi = d["max"][run_order, tech_idx] - total
    err = np.clip(np.array([err_lo, err_hi]), 0, None)
    ax.errorbar(x_positions_MARL, height, yerr=err,
                fmt='none', ecolor='black', capsize=0)

    # P25-P75 whisker (with caps)
    err_lo = total - d["p25"][run_order, tech_idx]
    err_hi = d["p75"][run_order, tech_idx] - total
    err = np.clip(np.array([err_lo, err_hi]), 0, None)
    ax.errorbar(x_positions_MARL, height, yerr=err,
                fmt='none', ecolor='black', capsize=5)

    # Group separators
    for pos in separator_positions:
        ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)

    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    if show_xlabels:
        ax.set_xticks(x_positions_MARL)
        ax.set_xticklabels(["ND", "D"] * 4, ha="center", fontsize=7)
    else:
        ax.set_xticks([])

    if show_group_labels:
        for gi, glab in enumerate(group_labels):
            ax.text(group_positions[gi], -0.2, glab,
                    ha='center', va='top',
                    fontsize=8, fontweight='bold',
                    transform=ax.get_xaxis_transform())
                    
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    if show_ylabel:
        ax.set_ylabel("[GW]", fontsize=9)


# -----------------------------------------------------------------------------
# Figure layout
#
# Outer grid: 4 rows x 2 cols (one cell per technology, matches original).
# Each outer cell holds a nested 1x3 grid: 2030 | 2035 | 2040, sharing y-axis.
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 8))

outer = gridspec.GridSpec(
    n_outer_rows, n_outer_cols,
    figure=fig,
    hspace=0.4,   # vertical space between technology rows
    wspace=0.1,   # horizontal space between the two technology columns
    left=0.06, right=0.99, top=0.95, bottom=0.05,
)

# Track all axes for legend extraction later
all_axes = []

for tech_idx, title, orow, ocol in panel_spec:
    # Inner 1x3 grid for the three years; shared y-axis, tight spacing.
    inner = gridspec.GridSpecFromSubplotSpec(
        1, n_years,
        subplot_spec=outer[orow, ocol],
        wspace=0.08,
    )

    # Bottom-most technology in each column gets group labels + x-labels
    is_bottom_in_column = (orow == n_outer_rows - 1)

    inner_axes = []
    for j, year in enumerate(years):
        if j == 0:
            ax = fig.add_subplot(inner[0, j])
        else:
            # Share y-axis with the first sub-panel of this technology
            ax = fig.add_subplot(inner[0, j], sharey=inner_axes[0])
            # Hide tick labels on the right two sub-panels
            plt.setp(ax.get_yticklabels(), visible=False)

        plot_panel(
            ax, data[year], tech_idx,
            show_xlabels=is_bottom_in_column,
            show_group_labels=is_bottom_in_column,
            show_ylabel=(j == 0),
        )

        # Year label sits as the small title of each sub-panel
        ax.set_title(str(year), fontsize=9)

        inner_axes.append(ax)
        all_axes.append(ax)

    # Technology title spans the three sub-panels: place it above them.
    bbox_left  = inner_axes[0].get_position()
    bbox_right = inner_axes[-1].get_position()
    x_center = 0.5 * (bbox_left.x0 + bbox_right.x1)
    y_top    = bbox_left.y1
    fig.text(
        x_center, y_top + 0.025, title,
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
    )

# Shared legend at the top
handles, labels = all_axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center', ncol=5,
    bbox_to_anchor=(0.5, 0.00),
    frameon=False, fontsize=10,
)

plt.savefig("plot_bars_capacities_2040_V6.pdf",
            format="pdf", bbox_inches="tight")
print("Saved plot_bars_capacities_2040_V6.pdf")