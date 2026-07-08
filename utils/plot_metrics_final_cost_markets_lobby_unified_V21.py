import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams.update({'font.size': 10})

# ============================================================================
# SHARED CONFIG
# ============================================================================

data_path = '/work/cmcc/jg24923/CVaR_58/'

"""
checkpoint_runs = [
    "28_CM_CfD_28", "29_CM_CfD_29", "11_CM_CfD_11", "12_CM_CfD_12",
    "30_CM_CfD_30", "31_CM_CfD_31", "62_CM_CfD_62", "63_CM_CfD_63",
]
"""

checkpoint_runs = [
    "62_CM_CfD_62", "63_CM_CfD_63", "30_CM_CfD_30", "31_CM_CfD_31",
    "11_CM_CfD_11", "12_CM_CfD_12", "28_CM_CfD_28", "29_CM_CfD_29",
]

number_runs  = len(checkpoint_runs)
length_sim   = 102

T_START_AGG = 6
T_END_AGG   = 102

WINDOWS = [
    ('Short\n(2025–2026)',  6,  18,  2),
    ('Mid\n(2027–2034)',   18,  66,  8),
    ('Long\n(2035–2040)',  66, 102,  5),
]

x_positions         = list(np.arange(0, 11))
x_positions_MARL    = [0, 1, 3, 4, 6, 7, 9, 10]
separator_positions = [2, 5, 8]
group_positions     = [0.5, 3.5, 6.5, 9.5]
# group_labels        = ["H+", "H", "H-", "CRM"]
group_labels        = ["CRM", "H-", "H", "H+"]
run_order_names     = ["ND", "D", "", "ND", "D", "", "ND", "D", "", "ND", "D"]
bar_width           = 0.75

# ── COLORS ─────────────────────────────────────────────────────────────────
# Emissions bars: green (COLOR_A). ND solid, D white + hatched.
COLOR_A           = '#0F6E56'
COLOR_B           = '#0F6E56'
# EENS bars: distinct color so the security metric reads separately from
# emissions. Same ND-solid / D-hatched pattern.
COLOR_EENS        = '#7F2A8C'

colors            = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5',
                     '#00A2E9', '#003F5C', '#00E5FF', '#00B89F']
color_existing    = colors[0]
color_merchant    = colors[2]
color_cm          = '#007EC5'
color_CfD         = '#00A2E9'
color_flexibility = '#003F5C'
color_scarcity    = '#FF6B35'
color_tax_return  = 'lightgrey'
color_others      = '#7F7F7F'

# Per-bar style for emissions panel. Indices alternate ND, D, ND, D, …
# ND → solid fill in COLOR_A
# D  → white fill + hatched in COLOR_A
bar_face_colors   = [COLOR_A, 'white'] * 4
bar_edge_colors   = [COLOR_A, COLOR_A] * 4
bar_hatches       = ['',      '////']  * 4

# Same pattern, but in COLOR_EENS for the EENS panel.
bar_face_colors_eens = [COLOR_EENS, 'white']    * 4
bar_edge_colors_eens = [COLOR_EENS, COLOR_EENS] * 4
bar_hatches_eens     = ['',         '////']     * 4

# ============================================================================
# DATA LOADING
# ============================================================================

prices_all   = np.zeros([number_runs, length_sim])
ref_all      = np.zeros([number_runs, length_sim])
cm_all       = np.zeros([number_runs, length_sim])
cfd_all      = np.zeros([number_runs, length_sim])
flex_all     = np.zeros([number_runs, length_sim])
scarcity_all = np.zeros([number_runs, length_sim])
tax_all      = np.zeros([number_runs, length_sim])
merchant_all = np.zeros([number_runs, length_sim])
existing_all = np.zeros([number_runs, length_sim])

# cost_raw_all[i]  : (steps, n_seeds) — per-seed total system cost col 0
# demand_all[i]     : (steps, 1)       — average demand per timestep (constant across seeds)
cost_raw_all  = []
emiss_raw_all = []
ens_raw_all   = []
demand_all    = []

for i in range(number_runs):
    folder = f"{data_path}{checkpoint_runs[i]}"

    # Per-seed total system cost (cost_spot_other_markets_0.csv)
    c0_raw = np.array(pd.read_csv(
        os.path.join(folder, 'cost_spot_other_markets_0.csv')).iloc[:, 1:])
    cost_raw_all.append(c0_raw[:length_sim, :])

    # Average demand trajectory — shape (steps, 1) for broadcasting
    d_raw = np.array(pd.read_csv(
        os.path.join(folder, 'average_demand.csv')).iloc[:, 1:])
    demand_all.append(d_raw[:length_sim, 0:1])          # (steps, 1)

    # Mean normalised cost -> trajectory for stacked bar decomposition
    prices_all[i, :] = np.mean(c0_raw[:length_sim, :] / d_raw[:length_sim, 0:1], axis=1)

    e_raw = np.array(pd.read_csv(os.path.join(folder, 'CO2_emissions.csv')).iloc[:, 1:])
    emiss_raw_all.append(e_raw[:length_sim, :])

    ens_raw = np.array(pd.read_csv(
        os.path.join(folder, 'energy_not_served.csv')).iloc[:, 1:])
    ens_raw_all.append(ens_raw[:length_sim, :])

    c = np.array(pd.read_csv(
        os.path.join(folder, 'cost_spot_other_markets.csv')).iloc[:, 1:])
    ref_all[i, :]      = c[:length_sim, 0]
    cm_all[i, :]       = c[:length_sim, 1]
    cfd_all[i, :]      = c[:length_sim, 2]
    flex_all[i, :]     = c[:length_sim, 3]
    scarcity_all[i, :] = c[:length_sim, 4]
    tax_all[i, :]      = c[:length_sim, 5]
    merchant_all[i, :] = c[:length_sim, 6]
    existing_all[i, :] = c[:length_sim, 7]

# ============================================================================
# AGGREGATE STATS
# ============================================================================

N_YEARS_AGG = 15

def agg_stats(s, e):
    cost_comp = {}
    cost_pct  = {}
    emiss_pct = {}
    eens_pct  = {}

    for i in range(number_runs):
        pr  = prices_all[i, s:e]
        ref = ref_all[i, s:e]

        cm_pu       = (cm_all[i, s:e]       / ref) * pr
        cfd_pu      = (cfd_all[i, s:e]      / ref) * pr
        flex_pu     = (flex_all[i, s:e]     / ref) * pr
        scarcity_pu = (scarcity_all[i, s:e] / ref) * pr
        tax_pu      = (tax_all[i, s:e]      / ref) * pr
        merchant_pu = (merchant_all[i, s:e] / ref) * pr
        existing_pu = (existing_all[i, s:e] / ref) * pr
        named_pu    = (cm_pu + cfd_pu + flex_pu + scarcity_pu
                       + tax_pu + merchant_pu + existing_pu)

        cost_comp[i] = dict(
            cm=np.mean(cm_pu), cfd=np.mean(cfd_pu),
            flex=np.mean(flex_pu), scarcity=np.mean(scarcity_pu),
            tax=np.mean(tax_pu), merchant=np.mean(merchant_pu),
            existing=np.mean(existing_pu),
            others=np.mean(pr - named_pu),
        )

        # Per-seed mean total system cost normalised by average demand [EUR/MWh].
        # cost_raw_all[i][s:e, :] : (steps, seeds)
        # demand_all[i][s:e, :]   : (steps, 1)  -> broadcasts across seeds
        per_seed_cost = np.mean(
            cost_raw_all[i][s:e, :] / demand_all[i][s:e, :], axis=0)
        cost_pct[i] = dict(
            mean=np.mean(per_seed_cost),
            p25=np.percentile(per_seed_cost, 25),
            p75=np.percentile(per_seed_cost, 75),
            p5=np.percentile(per_seed_cost,  5),
            p95=np.percentile(per_seed_cost, 95),
        )

        n_years = (e - s) / 6
        per_seed_emiss = np.sum(emiss_raw_all[i][s:e, :], axis=0) / (n_years * 1e6)
        emiss_pct[i] = dict(
            mean=np.mean(per_seed_emiss),
            p25=np.percentile(per_seed_emiss, 25),
            p75=np.percentile(per_seed_emiss, 75),
            p5=np.percentile(per_seed_emiss,  5),
            p95=np.percentile(per_seed_emiss, 95),
        )

        # ENS normalised by average demand at each timestep (mirrors metric_ens).
        # ens_raw_all[i][s:e, :] : (steps, seeds)
        # demand_all[i][s:e, :]  : (steps, 1) -> broadcasts across seeds
        # mean_t( ens[t,s] * 61.2 / demand[t] ) * 100  => % of average demand
        per_seed_eens = (np.mean(
            ens_raw_all[i][s:e, :] * 61.2 / demand_all[i][s:e, :], axis=0) * 100)
        eens_pct[i] = dict(
            mean=np.mean(per_seed_eens),
            p25=np.percentile(per_seed_eens, 25),
            p75=np.percentile(per_seed_eens, 75),
            p5=np.percentile(per_seed_eens,  5),
            p95=np.percentile(per_seed_eens, 95),
        )

    return cost_comp, cost_pct, emiss_pct, eens_pct


agg_cc, agg_cp, agg_ep, agg_en = agg_stats(T_START_AGG, T_END_AGG)

window_data = []
for label, s, e, n_years in WINDOWS:
    cc, cp, ep, _ = agg_stats(s, e)
    window_data.append((cc, cp, ep))

# ============================================================================
# DRAWING HELPERS
# ============================================================================

def _sep_and_xlabels(ax, show_group_labels=False, margin = -0.1):
    for sp in separator_positions:
        ax.axvline(x=sp, color='grey', linestyle='--', linewidth=0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(run_order_names, ha='center', fontsize=10)
    if show_group_labels:
        for gl, gp in zip(group_labels, group_positions):
            ax.text(gp, margin, gl,
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    transform=ax.get_xaxis_transform())


def _draw_nd_d_bars(ax, heights, face_colors=None, edge_colors=None, hatches=None):
    """Draw the 8 emission/EENS bars with the ND-solid / D-hatched pattern.

    Defaults to the emissions-panel green styling; pass alternate lists to
    re-use the same pattern in a different colour (e.g. the EENS panel)."""
    if face_colors is None:
        face_colors = bar_face_colors
    if edge_colors is None:
        edge_colors = bar_edge_colors
    if hatches is None:
        hatches = bar_hatches
    for x, h, fc, ec, ht in zip(x_positions_MARL, heights,
                                face_colors, edge_colors, hatches):
        ax.bar(x, h, width=bar_width,
               color=fc, edgecolor=ec, hatch=ht, linewidth=1.0)


def draw_cost_panel(ax, cost_comp, cost_pct,
                    show_ylabel=False, show_legend=False,
                    col_title=None, show_yticklabels=True,
                    show_xticklabels=False, show_group_labels=False,
                    ylim=(0, 125), margin = -0.15):
    cm_pos  = np.array([max(cost_comp[i]['cm'],  0) for i in range(number_runs)])
    cm_neg  = np.array([min(cost_comp[i]['cm'],  0) for i in range(number_runs)])
    cfd_pos = np.array([max(cost_comp[i]['cfd'], 0) for i in range(number_runs)])
    cfd_neg = np.array([min(cost_comp[i]['cfd'], 0) for i in range(number_runs)])

    sc  = np.array([cost_comp[i]['scarcity'] for i in range(number_runs)])
    fl  = np.array([cost_comp[i]['flex']     for i in range(number_runs)])
    tx  = np.array([cost_comp[i]['tax']      for i in range(number_runs)])
    me  = np.array([cost_comp[i]['merchant'] for i in range(number_runs)])
    ex  = np.array([cost_comp[i]['existing'] for i in range(number_runs)])
    ot  = np.array([cost_comp[i]['others']   for i in range(number_runs)])
    ot_pos = np.maximum(ot, 0)
    ot_neg = np.minimum(ot, 0)

    ax.bar(x_positions_MARL, sc,     width=bar_width,
           color=color_scarcity, edgecolor=color_scarcity, label='EEN')
    b2 = sc
    ax.bar(x_positions_MARL, cm_pos, width=bar_width, bottom=b2,
           color='white', hatch='xx', edgecolor=color_cm, label='CRM')
    b3 = b2 + cm_pos
    ax.bar(x_positions_MARL, cfd_pos, width=bar_width, bottom=b3,
           color=color_CfD, edgecolor=color_CfD, label='CfD')
    b4 = b3 + cfd_pos
    ax.bar(x_positions_MARL, fl,  width=bar_width, bottom=b4,
           color='white', hatch='oo', edgecolor=color_flexibility, label='Flexibility')
    b5 = b4 + fl
    ax.bar(x_positions_MARL, tx,  width=bar_width, bottom=b5,
           color=color_tax_return, edgecolor='black', hatch='////', label='ETS recovery')
    b6 = b5 + tx
    ax.bar(x_positions_MARL, ex,  width=bar_width, bottom=b6,
           color=color_existing, edgecolor=color_existing, label='Existing')
    b7 = b6 + ex
    ax.bar(x_positions_MARL, me,  width=bar_width, bottom=b7,
           color=color_merchant, edgecolor=color_merchant, label='Merchant')
    b8 = b7 + me
    ax.bar(x_positions_MARL, ot_pos, width=bar_width, bottom=b8,
           color=color_others, edgecolor=color_others, label='Others')

    ax.bar(x_positions_MARL, cfd_neg, width=bar_width,
           color=color_CfD, edgecolor=color_CfD, alpha=0.5)
    ax.bar(x_positions_MARL, cm_neg, width=bar_width, bottom=cfd_neg,
           color='white', hatch='xx', edgecolor=color_cm, alpha=0.5)
    ax.bar(x_positions_MARL, ot_neg, width=bar_width, bottom=cfd_neg + cm_neg,
           color=color_others, edgecolor=color_others, alpha=0.5)

    m   = np.array([cost_pct[i]['mean'] for i in range(number_runs)])
    p25 = np.array([cost_pct[i]['p25']  for i in range(number_runs)])
    p75 = np.array([cost_pct[i]['p75']  for i in range(number_runs)])
    p5  = np.array([cost_pct[i]['p5']   for i in range(number_runs)])
    p95 = np.array([cost_pct[i]['p95']  for i in range(number_runs)])

    ax.errorbar(x_positions_MARL, m,
                yerr=np.clip([m - p5,  p95 - m], 0, None),
                fmt='none', ecolor='black', capsize=0, linewidth=0.8)
    ax.errorbar(x_positions_MARL, m,
                yerr=np.clip([m - p25, p75 - m], 0, None),
                fmt='none', ecolor='black', capsize=5, linewidth=1.5)

    ax.axhline(y=0, color='black', linewidth=0.6)
    _sep_and_xlabels(ax, show_group_labels=show_group_labels, margin = margin)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_ylim(*ylim)

    if show_ylabel:
        ax.set_ylabel('[€/MWh]', fontsize=11)
    if col_title:
        ax.set_title(col_title, fontsize=11, fontweight='bold', pad=5)
    if show_legend:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9,
                  ncol=4, columnspacing=0.7, handlelength=2.0)
    if not show_yticklabels:
        ax.set_yticklabels([])
    if not show_xticklabels:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)


def draw_emiss_panel(ax, emiss_pct,
                     show_ylabel=False, show_group_labels=False,
                     show_yticklabels=True, show_xticklabels=True,
                     ylim=(0, 100), col_title=None, margin = -0.15):
    m   = np.array([emiss_pct[i]['mean'] for i in range(number_runs)])
    p25 = np.array([emiss_pct[i]['p25']  for i in range(number_runs)])
    p75 = np.array([emiss_pct[i]['p75']  for i in range(number_runs)])
    p5  = np.array([emiss_pct[i]['p5']   for i in range(number_runs)])
    p95 = np.array([emiss_pct[i]['p95']  for i in range(number_runs)])

    _draw_nd_d_bars(ax, m)

    ax.errorbar(x_positions_MARL, m,
                yerr=np.clip([m - p5,  p95 - m], 0, None),
                fmt='none', ecolor='black', capsize=0, linewidth=0.8)
    ax.errorbar(x_positions_MARL, m,
                yerr=np.clip([m - p25, p75 - m], 0, None),
                fmt='none', ecolor='black', capsize=5, linewidth=1.5)

    ax.axhline(y=0, color='black', linewidth=0.6)
    _sep_and_xlabels(ax, show_group_labels=show_group_labels, margin = margin)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_ylim(*ylim)

    if show_ylabel:
        ax.set_ylabel('[MtCO$_2$/year]', fontsize=11)
    if col_title:
        ax.set_title(col_title, fontsize=11, fontweight='bold', pad=5)
    if not show_yticklabels:
        ax.set_yticklabels([])
    if not show_xticklabels:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)


def draw_eens_panel(ax, eens_pct,
                    show_ylabel=False, show_group_labels=False,
                    show_yticklabels=True, margin = -0.15):
    m   = np.array([eens_pct[i]['mean'] for i in range(number_runs)])
    p25 = np.array([eens_pct[i]['p25']  for i in range(number_runs)])
    p75 = np.array([eens_pct[i]['p75']  for i in range(number_runs)])
    p5  = np.array([eens_pct[i]['p5']   for i in range(number_runs)])
    p95 = np.array([eens_pct[i]['p95']  for i in range(number_runs)])

    _draw_nd_d_bars(ax, m,
                    face_colors=bar_face_colors_eens,
                    edge_colors=bar_edge_colors_eens,
                    hatches=bar_hatches_eens)

    ax.axhline(y=0, color='black', linewidth=0.6)
    _sep_and_xlabels(ax, show_group_labels=show_group_labels, margin = margin)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    if show_ylabel:
        ax.set_ylabel('[%]', fontsize=11)
    if not show_yticklabels:
        ax.set_yticklabels([])

# ============================================================================
# FIGURE LAYOUT
# ============================================================================

fig = plt.figure(figsize=(14, 10))

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

height_ratios = [2, 0.45, 1, 1]

outer = GridSpec(
    nrows=4, ncols=1,
    figure=fig,
    height_ratios=height_ratios,
    hspace=0.15,
    top=0.95, bottom=0.08, left=0.07, right=0.98,
)

gs_row0 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.1)
gs_row2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.08)
gs_row3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.08)

ax_agg_cost  = fig.add_subplot(gs_row0[0, 0])
ax_agg_emiss = fig.add_subplot(gs_row0[0, 1])

ax_win_cost  = [fig.add_subplot(gs_row2[0, col]) for col in range(3)]
ax_win_emiss = [fig.add_subplot(gs_row3[0, col]) for col in range(3)]

# ============================================================================
# ROW 0: AGGREGATE
# ============================================================================

draw_cost_panel(
    ax_agg_cost, agg_cc, agg_cp,
    show_ylabel=True,
    show_legend=True,
    col_title='Total Unitary Cost',
    show_yticklabels=True,
    show_xticklabels=True,
    show_group_labels=True,
    ylim=(0, 125), margin = -0.1,
)
ax_agg_cost.set_ylabel('[€/MWh]', fontsize=11)

draw_emiss_panel(
    ax_agg_emiss, agg_ep,
    show_ylabel=True,
    show_group_labels=True,
    show_yticklabels=True,
    show_xticklabels=True,
    col_title='Yearly Emissions',
    ylim=(0, 70) , margin = -0.1,
)
ax_agg_emiss.set_ylabel('[MtCO$_2$/year]', fontsize=11)

# ============================================================================
# ROWS 2–3: TIME-WINDOWED PANELS
# ============================================================================

col_titles = [label for label, *_ in WINDOWS]

for col, ((label, s, e, n_years), (cc, cp, ep)) in enumerate(
        zip(WINDOWS, window_data)):

    draw_cost_panel(
        ax_win_cost[col], cc, cp,
        show_ylabel      = (col == 0),
        show_legend      = False,
        col_title        = col_titles[col],
        show_yticklabels = (col == 0),
        show_xticklabels = False,
        ylim=(0, 125), margin = -0.15,
    )
    if col == 0:
        ax_win_cost[col].set_ylabel('Total Unitary cost\n[€/MWh]', fontsize=11)

    draw_emiss_panel(
        ax_win_emiss[col], ep,
        show_ylabel       = (col == 0),
        show_group_labels = True,
        show_yticklabels  = (col == 0),
        show_xticklabels  = True,
        ylim=(0, 70), margin = -0.15,
    )
    if col == 0:
        ax_win_emiss[col].set_ylabel('Yearly Emissions\n[MtCO$_2$/year]', fontsize=11)

# ============================================================================
# PANEL LABELS (d)–(i) for windowed panels
# ============================================================================
"""
panel_letters = ['d', 'e', 'f', 'g', 'h', 'i']
windowed_axes = ax_win_cost + ax_win_emiss
for ax, letter in zip(windowed_axes, panel_letters):
    ax.text(-0.04, 1.04, f'({letter})',
            transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='right')
"""
# ============================================================================
# SHARED COST-PANEL LEGEND (windowed row only)
# ============================================================================
cost_handles, cost_labels = ax_agg_cost.get_legend_handles_labels()

win_left_bbox  = ax_win_cost[0].get_position()
win_right_bbox = ax_win_cost[2].get_position()
win_x_center   = 0.5 * (win_left_bbox.x0 + win_right_bbox.x1)
win_y_top      = win_left_bbox.y1

fig.legend(
    cost_handles, cost_labels,
    loc='lower center',
    bbox_to_anchor=(win_x_center, win_y_top + 0.04),
    ncol=8,
    frameon=False,
    fontsize=10,
    columnspacing=1.5,
    handlelength=2.0,
)

# ============================================================================
# SAVE
# ============================================================================

fig.savefig("plot_metrics_final_cost_markets_lobby_unified_V21.pdf",
            format="pdf", bbox_inches="tight")
print("Saved: plot_metrics_final_cost_markets_lobby_unified_V21.pdf")