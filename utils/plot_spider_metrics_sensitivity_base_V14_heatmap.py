import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams.update({'font.size': 10})

# ============================================================================
# DATA LOADING
# ============================================================================

data_path = '/work/cmcc/jg24923/CVaR_58/'

number_runs = 12

"""
checkpoint_runs = [
    "28_CM_CfD_28", "20_CM_CfD_20", "29_CM_CfD_29", # Hybrid+: ND, UD, D
    "11_CM_CfD_11", "19_CM_CfD_19", "12_CM_CfD_12", # Hybrid:  ND, UD, D
    "30_CM_CfD_30", "36_CM_CfD_36", "31_CM_CfD_31", # Hybrid-: ND, UD, D
    "62_CM_CfD_62", "64_CM_CfD_64", "63_CM_CfD_63", # CRM:     ND, UD, D
]

checkpoint_runs_no_price_schocks = [
    "43_CM_CfD_43", "45_CM_CfD_45", "44_CM_CfD_44", # Hybrid+: ND, UD, D
    "37_CM_CfD_37", "39_CM_CfD_39", "38_CM_CfD_38", # Hybrid:  ND, UD, D
    "46_CM_CfD_46", "48_CM_CfD_48", "47_CM_CfD_47", # Hybrid-: ND, UD, D
    "69_CM_CfD_69", "71_CM_CfD_71", "70_CM_CfD_70", # CRM:     ND, UD, D
]
"""

checkpoint_runs = [
    "62_CM_CfD_62", "64_CM_CfD_64", "63_CM_CfD_63", # CRM:     ND, UD, D
    "30_CM_CfD_30", "36_CM_CfD_36", "31_CM_CfD_31", # Hybrid-: ND, UD, D
    "11_CM_CfD_11", "19_CM_CfD_19", "12_CM_CfD_12", # Hybrid:  ND, UD, D
    "28_CM_CfD_28", "20_CM_CfD_20", "29_CM_CfD_29", # Hybrid+: ND, UD, D
]

checkpoint_runs_no_price_schocks = [
    "69_CM_CfD_69", "71_CM_CfD_71", "70_CM_CfD_70", # CRM:     ND, UD, D
    "46_CM_CfD_46", "48_CM_CfD_48", "47_CM_CfD_47", # Hybrid-: ND, UD, D
    "37_CM_CfD_37", "39_CM_CfD_39", "38_CM_CfD_38", # Hybrid:  ND, UD, D
    "43_CM_CfD_43", "45_CM_CfD_45", "44_CM_CfD_44", # Hybrid+: ND, UD, D
]

T_START = 6
T_END   = 96

T_START_VOLATILITY = 24 * 6 * 1
T_END_VOLATILITY   = 24 * 6 * 16
N_YEARS = 15

is_incumbent = [0, 1, 2, 3, 4, 5, 6, 7]
is_entrant   = [8, 9, 10, 11, 12, 13, 14, 15]
is_RES       = [2, 11, 12, 13, 14, 15]
is_FF        = [3, 4, 8, 9, 10]


def load_total_cost(run_names):
    """Return the per-run net total-cost scalar (mean over T_START:T_END)
    for each run in `run_names`.

    Definition: cost_spot_other_markets.csv col 0 (total)
                − col 4 (scarcity) − col 5 (carbon tax return).
    Same `total_cost` expression used in the main metric loop."""
    out = np.zeros(len(run_names))
    for j, name in enumerate(run_names):
        folder = f"{data_path}{name}"
        m = np.array(pd.read_csv(
            os.path.join(folder, 'cost_spot_other_markets.csv')).iloc[:, 1:])

        """
        out[j] = np.mean(m[T_START:T_END, 0]
                         - m[T_START:T_END, 4]
                         - m[T_START:T_END, 5])

        """
        out[j] = np.mean(m[T_START:T_END, 0])

    return out


prices_raw            = []
emissions_raw         = []
volatility_raw        = []
volatility_net_raw    = []
res_curtailed_raw     = []
ENS_raw               = []
profits_incumbent_raw = []
profits_entrants_raw  = []
profits_RES_raw       = []
profits_FF_raw        = []
incumbent_ratio_raw   = []
res_share_raw         = []
storage_share_raw     = []
merchant_ratio_raw    = []
auction_dependance_raw = []
existing_cost_raw     = []
total_cost_shock      = []   # per-run total_cost, shock runs

for j in range(number_runs):
    folder = f"{data_path}{checkpoint_runs[j]}"

    prices = np.array(pd.read_csv(
        os.path.join(folder, 'prices.csv')).iloc[T_START:T_END, 1:])
    prices_raw.append(np.mean(prices))

    emissions = np.array(pd.read_csv(
        os.path.join(folder, 'CO2_emissions.csv')).iloc[T_START:T_END, 1:])
    emissions_raw.append(np.mean(emissions))

    profits_temp = np.array(pd.read_csv(
        os.path.join(folder, 'Reward_Penalty_agent.csv')).iloc[:, 1:])
    profits_incumbent_raw.append(np.mean(
        sum(profits_temp[i, :] for i in is_incumbent)))
    profits_RES_raw.append(np.mean(
        sum(profits_temp[i, :] for i in is_RES)))
    profits_FF_raw.append(np.mean(
        sum(profits_temp[i, :] for i in is_FF)))
    profits_entrants_raw.append(np.mean(
        sum(profits_temp[i, :] for i in is_entrant)))

    st_prices = np.array(pd.read_csv(
        os.path.join(folder, 'prices_short_term.csv')
    ).iloc[T_START_VOLATILITY:T_END_VOLATILITY, 1:])
    volatility_raw.append(np.std(st_prices))

    st_prices_net = np.array(pd.read_csv(
        os.path.join(folder, 'prices_short_term_net.csv')
    ).iloc[T_START_VOLATILITY:T_END_VOLATILITY, 1:])
    volatility_net_raw.append(np.std(st_prices_net))

    res_curtailed = np.array(pd.read_csv(
        os.path.join(folder, 'RES_curtailed.csv')).iloc[:, 1:])
    res_curtailed_raw.append(np.mean(res_curtailed))

    res_share = np.array(pd.read_csv(
        os.path.join(folder, 'res_share.csv')).iloc[T_START:T_END, 1:])
    res_share_raw.append(np.mean(res_share))

    ENS = np.array(pd.read_csv(
        os.path.join(folder, 'energy_not_served.csv')).iloc[T_START:T_END, 1:])
    ENS_raw.append(np.mean(ENS))

    cap_2040 = np.array(pd.read_csv(
        os.path.join(folder, 'capacity_2040_agent.csv')).iloc[:, 1:])
    total_cap     = cap_2040.sum(axis=0)
    incumbent_cap = sum(cap_2040[i, :] for i in is_incumbent)
    incumbent_ratio_raw.append(np.mean(incumbent_cap / (total_cap)))

    storage_share = np.array(pd.read_csv(
        os.path.join(folder, 'storage_share.csv')).iloc[:, 1:]) * 100
    storage_share_raw.append(np.mean(storage_share))

    mkt_contrib = np.array(pd.read_csv(
        os.path.join(folder, 'cost_spot_other_markets.csv')).iloc[:, 1:])

    """
    total_cost = np.mean(mkt_contrib[T_START:T_END, 0]
                         - mkt_contrib[T_START:T_END, 4]
                         - mkt_contrib[T_START:T_END, 5])
    """
    total_cost = np.mean(mkt_contrib[T_START:T_END, 0])

    total_cost_shock.append(total_cost)

    merchant_cost = np.mean(mkt_contrib[T_START:T_END, 6])
    merchant_ratio_raw.append(merchant_cost / total_cost)

    existing_cost = np.mean(mkt_contrib[T_START:T_END, 7])
    existing_cost_raw.append(existing_cost / total_cost)

    auction_dependance = np.mean(mkt_contrib[T_START:T_END, 1]
                                 + mkt_contrib[T_START:T_END, 2]
                                 + mkt_contrib[T_START:T_END, 3])
    auction_dependance_raw.append(auction_dependance / total_cost)


# ── Shock-resilience indicator: total_cost(shock) / total_cost(no_shock) ────
total_cost_shock    = np.array(total_cost_shock)
total_cost_no_shock = load_total_cost(checkpoint_runs_no_price_schocks)
shock_resilience_raw = (-total_cost_no_shock + total_cost_shock)/total_cost_no_shock


# Convert remaining lists to arrays
prices_raw             = np.array(prices_raw)
emissions_raw          = np.array(emissions_raw)
volatility_raw         = np.array(volatility_raw)
volatility_net_raw     = np.array(volatility_net_raw)
res_curtailed_raw      = np.array(res_curtailed_raw)
ENS_raw                = np.array(ENS_raw)
profits_incumbent_raw  = np.array(profits_incumbent_raw)
profits_RES_raw        = np.array(profits_RES_raw)
profits_FF_raw         = np.array(profits_FF_raw)
incumbent_ratio_raw    = np.array(incumbent_ratio_raw)
profits_entrants_raw   = np.array(profits_entrants_raw)
res_share_raw          = np.array(res_share_raw)
storage_share_raw      = np.array(storage_share_raw)
merchant_ratio_raw     = np.array(merchant_ratio_raw)
auction_dependance_raw = np.array(auction_dependance_raw)
existing_cost_raw      = np.array(existing_cost_raw)

# ============================================================================
# NORMALISATION
# ============================================================================

def normalise(raw, n_markets=4, n_scenarios=3, per_market=False):
    out = np.empty_like(raw, dtype=float)
    if per_market:
        for m in range(n_markets):
            sl      = slice(m * n_scenarios, (m + 1) * n_scenarios)
            ref     = np.mean(raw[sl])
            denom   = np.abs(ref) if np.abs(ref) > 1e-12 else 1.0
            out[sl] = (raw[sl] - ref) / denom
    else:
        ref   = np.mean(raw)
        denom = np.abs(ref) if np.abs(ref) > 1e-12 else 1.0
        out   = (raw - ref) / denom
    return out

metrics_raw = [
    volatility_raw,         # 0
    volatility_net_raw,     # 1
    shock_resilience_raw,   # 2  ← new, in Volatility group
    profits_incumbent_raw,  # 3
    profits_entrants_raw,   # 4
    merchant_ratio_raw,     # 5
    existing_cost_raw,      # 6
    auction_dependance_raw, # 7
    ENS_raw,                # 8
]

metrics_norm = [normalise(m, per_market=False) for m in metrics_raw]
heatmap_data = np.column_stack(metrics_norm)   # shape (12, 9)

# ============================================================================
# HEATMAP — 2×2 grid
# ============================================================================

objectives = [
    'Wholesale\nVolatility', 'Total Cost\nVolatility', 'Gas Shock\nVulnerability',  # group: Volatility
    'Profit\nIncumbents',    'Profit\nEntrants',         # group: Profits
    'Merchant\nShare',       'Existing\nShare',          # group: Market contributions
    'Mechanism\nShare',                                  # group: Market contributions (cont.)
    'Energy not\nServed',                                # group: Security
]

n_objectives    = len(objectives)
scenario_labels = ['ND', 'UD', 'D']
n_scenarios     = len(scenario_labels)

# market_labels = [' H+ ', '  H  ', '   H-   ', '   CRM   ']
market_labels = ['   CRM   ', '   H-   ', '  H  ', ' H+ ']
market_colors = ['#156082', '#00E5FF', '#00B89F', '#7F7F7F']

# Column indices AFTER which a group divider is drawn (between col i and i+1)
# Volatility (0-2) | Profits (3-4) | Market contributions (5-7) | Security (8)
GROUP_DIVIDERS = [2, 4, 7]

# Group span definitions (used for both brackets and labels)
GROUP_SPANS = [
    ('Volatility',           0, 2),
    ('Profits',              3, 4),
    ('Market Contributions', 5, 7),
    ('Security',             8, 8),
]

# ── Asymmetric colour scale ──────────────────────────────────────────────────
vmin, vmax = -1.0, 2.25
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu_r
heatmap_data_clamped = np.clip(heatmap_data, vmin, vmax)

# ── Perceived-luminance text colour helper ───────────────────────────────────
def cell_text_color(val, norm, cmap, lum_thresh=0.65):
    r, g, b, _ = cmap(norm(val))
    luminance   = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return '#222222' if luminance > lum_thresh else 'white'

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(20, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.15)

for m, (market, ax, market_color) in enumerate(
        zip(market_labels, axes.flat, market_colors)):

    market_data = heatmap_data_clamped[m * n_scenarios : (m + 1) * n_scenarios]

    ax.imshow(market_data, cmap=cmap, norm=norm,
              aspect='auto', interpolation='nearest')

    # ── Subtle cell grid lines ────────────────────────────────────────────
    for x in np.arange(-0.5, n_objectives, 1):
        ax.axvline(x, color='white', linewidth=0.5, alpha=0.5)
    for y in np.arange(-0.5, n_scenarios, 1):
        ax.axhline(y, color='white', linewidth=0.5, alpha=0.5)

    # ── Group divider lines (bold black, spanning full cell height) ───────
    for div_col in GROUP_DIVIDERS:
        ax.axvline(div_col + 0.5, color='black', linewidth=2.0,
                   linestyle='-', zorder=4)

    # ── Cell annotations (luminance-aware text colour) ────────────────────
    for row in range(n_scenarios):
        for col in range(n_objectives):
            val = market_data[row, col]
            txt_color = cell_text_color(val, norm, cmap)
            ax.text(col, row, f'{val:+.2f}',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    fontfamily='Nimbus Sans', color=txt_color)

    # ── Group brackets + labels above the top-row panels ─────────────────
    if m < 2:
        for label, col_start, col_end in GROUP_SPANS:
            ax.annotate('', xy=(col_end + 0.45, 1.22),
                        xytext=(col_start - 0.45, 1.22),
                        xycoords=('data', 'axes fraction'),
                        textcoords=('data', 'axes fraction'),
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', color='#444444',
                                        lw=1.2))
            x_center = (col_start + col_end) / 2.0
            ax.text(x_center, 1.25, label,
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    fontfamily='Nimbus Sans', color='#444444',
                    clip_on=False)

    # ── Y-axis ────────────────────────────────────────────────────────────
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_labels, fontsize=9.5,
                       fontfamily='Nimbus Sans', fontweight='bold')
    ax.text(-0.1, 0.5, market,
            transform=ax.transAxes,
            fontsize=11, fontweight='bold', fontfamily='Nimbus Sans',
            color='white', va='center', ha='center', rotation=90,
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor=market_color, edgecolor='none'))

    # ── X-axis: labels only on top row ───────────────────────────────────
    if m < 2:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xticks(range(n_objectives))
        ax.set_xticklabels(objectives, fontsize=9, fontweight='bold',
                           fontfamily='Nimbus Sans', rotation=0,
                           ha='center', multialignment='center')
    else:
        ax.set_xticks(range(n_objectives))
        ax.set_xticklabels([])
        ax.tick_params(bottom=False)

    for spine in ax.spines.values():
        spine.set_edgecolor('#000000')
        spine.set_linewidth(1.2)
    ax.tick_params(left=False, top=False, bottom=False)

# Extra top margin so group labels/brackets don't get clipped
fig.subplots_adjust(top=0.82)

# ── Shared colorbar with explicit asymmetric ticks & zero anchor ──────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                    fraction=0.03, pad=0.08, aspect=50, shrink=0.7)

cbar_ticks = [-1.0, -0.5, 0.0, 0.75, 1.50, 2.25]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([('0' if t == 0 else f'{t:+.2f}') for t in cbar_ticks])
cbar.ax.tick_params(labelsize=9)

cbar.ax.axvline(0, color='black', linewidth=1.4, linestyle='--', alpha=0.75)
cbar.set_label('Relative deviation from global mean',
               fontsize=10, fontweight='bold', fontfamily='Nimbus Sans')

plt.savefig("plot_spider_metrics_sensitivity_base_V14_heatmap.pdf",
            format="pdf", bbox_inches="tight")
plt.show()