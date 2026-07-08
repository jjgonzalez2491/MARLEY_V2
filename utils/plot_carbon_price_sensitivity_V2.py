import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams.update({'font.size': 10})

# ============================================================================
# DATA LOADING
# ============================================================================

data_path = '/work/cmcc/jg24923/CVaR_58/'

carbon_price_steps = [70, 150, 230]            # €/tCO2 — Low, Mid, High
n_cp = len(carbon_price_steps)

checkpoint_runs = [
    "24_CM_CfD_24", "11_CM_CfD_11",  "26_CM_CfD_26",
    "25_CM_CfD_25", "12_CM_CfD_12",  "27_CM_CfD_27",
    "65_CM_CfD_65", "62_CM_CfD_62",  "67_CM_CfD_67",
    "66_CM_CfD_66", "63_CM_CfD_63",  "68_CM_CfD_68",
]

checkpoint_runs_no_price_schocks = [
    "49_CM_CfD_49", "37_CM_CfD_37",  "51_CM_CfD_51",
    "50_CM_CfD_50", "38_CM_CfD_38",  "52_CM_CfD_52",
    "72_CM_CfD_72", "69_CM_CfD_69",  "74_CM_CfD_74",
    "73_CM_CfD_73", "70_CM_CfD_70",  "75_CM_CfD_75",
]

configs = [
    ('H · ND',   'H',   'ND', '#156082', '-',  'o'),
    ('H · D',    'H',   'D',  '#156082', '--', 's'),
    ('CRM · ND', 'CRM', 'ND', '#00B89F', '-',  'o'),
    ('CRM · D',  'CRM', 'D',  '#00B89F', '--', 's'),
]
n_configs = len(configs)
number_runs = n_configs * n_cp

T_START = 6
T_END   = 96

is_incumbent = [0, 1, 2, 3, 4, 5, 6, 7]
is_entrant   = [8, 9, 10, 11, 12, 13, 14, 15]
is_RES       = [2, 11, 12, 13, 14, 15]
is_FF        = [3, 4, 8, 9, 10]


def load_total_cost(run_names):
    out = np.zeros(len(run_names))
    for j, name in enumerate(run_names):
        folder = f"{data_path}{name}"
        m = np.array(pd.read_csv(
            os.path.join(folder, 'cost_spot_other_markets.csv')).iloc[:, 1:])

        demand = np.array(pd.read_csv(
            os.path.join(folder, 'average_demand.csv')).iloc[T_START:T_END, 1:])

        demand_t = np.sum(demand, axis=1) if demand.ndim > 1 else demand

        out[j] = np.mean(m[T_START:T_END, 0]) / np.mean(demand_t)

    return out


weighted_price_raw     = []
emissions_raw          = []
merchant_ratio_raw     = []
existing_cost_raw      = []
auction_dependance_raw = []
total_cost_shock       = []

for j in range(number_runs):
    folder = f"{data_path}{checkpoint_runs[j]}"

    mkt_contrib = np.array(pd.read_csv(
        os.path.join(folder, 'cost_spot_other_markets.csv')).iloc[:, 1:])

    demand = np.array(pd.read_csv(
        os.path.join(folder, 'average_demand.csv')).iloc[T_START:T_END, 1:])

    cost_t = mkt_contrib[T_START:T_END, 0]
    demand_t = np.sum(demand, axis=1) if demand.ndim > 1 else demand

    weighted_price_raw.append(np.mean(cost_t) / np.mean(demand_t))

    cost_t_tmp = mkt_contrib[T_START:T_END, 0]

    emissions = np.array(pd.read_csv(
        os.path.join(folder, 'CO2_emissions.csv')).iloc[T_START:T_END, 1:])
    emissions_raw.append(np.mean(emissions * 6 / (1e6)))

    total_cost_tmp = np.mean(cost_t_tmp) 
    total_cost_shock.append(total_cost_tmp/ np.mean(demand_t))

    merchant_cost = np.mean(mkt_contrib[T_START:T_END, 6])
    merchant_ratio_raw.append(merchant_cost / total_cost_tmp)

    existing_cost = np.mean(mkt_contrib[T_START:T_END, 7])
    existing_cost_raw.append(existing_cost / total_cost_tmp)

    auction_dependance = np.mean(
        mkt_contrib[T_START:T_END, 1]
        + mkt_contrib[T_START:T_END, 2]
        + mkt_contrib[T_START:T_END, 3]
    )
    auction_dependance_raw.append(auction_dependance / total_cost_tmp)

total_cost_shock     = np.array(total_cost_shock)
total_cost_no_shock  = load_total_cost(checkpoint_runs_no_price_schocks)
shock_resilience_raw = (total_cost_shock - total_cost_no_shock)/total_cost_no_shock * 100

weighted_price_raw     = np.array(weighted_price_raw).reshape(n_configs, n_cp)
emissions_raw          = np.array(emissions_raw).reshape(n_configs, n_cp)
merchant_ratio_raw     = np.array(merchant_ratio_raw).reshape(n_configs, n_cp) * 100
existing_cost_raw      = np.array(existing_cost_raw).reshape(n_configs, n_cp) * 100
auction_dependance_raw = np.array(auction_dependance_raw).reshape(n_configs, n_cp) * 100
shock_resilience_raw   = shock_resilience_raw.reshape(n_configs, n_cp)

# ============================================================================
# PLOTTING
# ============================================================================

metrics = [
    (weighted_price_raw,     'Total Unitary costs',      '[€/MWh]'),
    (emissions_raw,          'Yearly CO$_2$ Emissions',  '[MtCO$_2$]'),
    (merchant_ratio_raw,     'Merchant Share',           '[%]'),
    (existing_cost_raw,      'Existing Share',           '[%]'),
    (auction_dependance_raw, 'Mechanism Share',     '[%]'),
    (shock_resilience_raw,   'Gas Shock Vulnerability',  '[%]'),
]

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(
    2, 12, figure=fig,
    height_ratios=[1.25, 1.0],
    hspace=0.42, wspace=2.4,
)

ax_top1 = fig.add_subplot(gs[0, 0:6])
ax_top2 = fig.add_subplot(gs[0, 6:12])

ax_bot1 = fig.add_subplot(gs[1, 0:3])
ax_bot2 = fig.add_subplot(gs[1, 3:6])
ax_bot3 = fig.add_subplot(gs[1, 6:9])
ax_bot4 = fig.add_subplot(gs[1, 9:12])

axes = [ax_top1, ax_top2, ax_bot1, ax_bot2, ax_bot3, ax_bot4]
top_axes = [ax_top1, ax_top2]
bot_axes = [ax_bot1, ax_bot2, ax_bot3, ax_bot4]

x = np.array(carbon_price_steps)

for ax, (data, title, ylabel) in zip(axes, metrics):
    for i, (label, market, decree, color, ls, marker) in enumerate(configs):
        ax.plot(
            x, data[i, :],
            label=label,
            color=color, linestyle=ls, marker=marker,
            linewidth=2.0, markersize=7,
            markerfacecolor=color if decree == 'ND' else 'white',
            markeredgecolor=color, markeredgewidth=1.8,
        )

    # Title intentionally NOT set via ax.set_title() — added later in figure
    # coordinates so all titles in a row share the same y-position.
    ax.set_xlabel('Carbon price [€/tCO$_2$]', fontsize=10,
                  fontfamily='Nimbus Sans')
    ax.set_ylabel(ylabel, fontsize=10, fontfamily='Nimbus Sans')

    ax.set_xticks(carbon_price_steps)
    ax.tick_params(axis='both', labelsize=9)

    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    for spine in ax.spines.values():
        spine.set_edgecolor('#000000')
        spine.set_linewidth(1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax_bot4.axhline(1.0, color='#7F7F7F', linestyle=':', linewidth=1.0,
                alpha=0.8, zorder=0)

# ── Shared legend ───────────────────────────────────────────────────────────
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center', ncol=4,
    bbox_to_anchor=(0.5, -0.02),
    frameon=False, fontsize=11,
    handlelength=2.5, columnspacing=2.5,
    prop={'family': 'Nimbus Sans', 'weight': 'bold', 'size': 10},
)

# ── Aligned per-row titles in figure coordinates ────────────────────────────
# Draw once layout is finalised so axes positions are accurate.
fig.canvas.draw()

TITLE_PAD = 0.02   # vertical offset (figure fraction) above the row's top edge

def _add_row_titles(row_axes, row_metrics):
    """Place titles at a single shared y-coordinate for the row."""
    # Highest top edge across the row (in figure coords) → shared baseline
    y_top = max(ax.get_position().y1 for ax in row_axes)
    y_title = y_top + TITLE_PAD
    for ax, (_, title, _) in zip(row_axes, row_metrics):
        pos = ax.get_position()
        x_center = 0.5 * (pos.x0 + pos.x1)
        fig.text(
            x_center, y_title, title,
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            fontfamily='Nimbus Sans',
        )

_add_row_titles(top_axes, metrics[:2])
_add_row_titles(bot_axes, metrics[2:])

plt.savefig("plot_carbon_price_sensitivity_V2.pdf",
            format="pdf", bbox_inches="tight")
plt.show()