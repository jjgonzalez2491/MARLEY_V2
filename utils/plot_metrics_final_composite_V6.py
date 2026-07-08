import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams.update({'font.size': 10})

# ============================================================================
# SHARED CONFIG
# ============================================================================

data_path = '/work/cmcc/jg24923/CVaR_58/'

# Six checkpoints, all H scenario:
#   first three  → IPPO
#   last three   → MAPPO
checkpoint_runs = [
    "12_CM_CfD_12", "76_CM_CfD_76", "77_CM_CfD_77",   # H + IPPO
    "83_CM_CfD_83", "84_CM_CfD_84", "85_CM_CfD_85",   # H + MAPPO
]

number_runs  = len(checkpoint_runs)
length_sim   = 96
T_START      = 6
T_END        = 102
n_tech       = 8

# Bar geometry: 3 bars, gap, 3 bars  →  positions 0,1,2, [sep at 3], 4,5,6
bar_positions       = [0, 1, 2, 4, 5, 6]
separator_positions = [3]
x_tick_positions    = bar_positions
x_tick_labels       = ["", "", "", "", "", ""]

method_positions = [1, 5]
method_labels    = ["IPPO", "MAPPO"]

LAYER_Y_METHOD = -0.08

run_order = np.arange(number_runs, dtype=np.int8)
bar_width = 0.75

# ── COLORS ────────────────────────────────────────────────────────────────
COLOR_A           = '#0F6E56'
COLOR_EENS        = '#7F2A8C'
color_spot        = '#00CAD2'
color_cm          = '#007EC5'
color_CfD         = '#00A2E9'
color_flexibility = '#003F5C'
color_scarcity    = '#FF6B35'
color_tax_return  = 'lightgrey'
colors = ['#156082', '#7F7F7F', '#00CAD2', '#007EC5',
          '#00A2E9', '#003F5C', '#00E5FF', '#00B89F']

bar_face_colors      = [COLOR_A]    * number_runs
bar_edge_colors      = [COLOR_A]    * number_runs
bar_hatches          = ['']         * number_runs

bar_face_colors_eens = [COLOR_EENS] * number_runs
bar_edge_colors_eens = [COLOR_EENS] * number_runs
bar_hatches_eens     = ['']         * number_runs


# ============================================================================
# HELPERS
# ============================================================================

def draw_xaxis_layers(ax):
    for pos, lbl in zip(method_positions, method_labels):
        ax.text(pos, LAYER_Y_METHOD, lbl,
                ha='center', va='top',
                fontsize=11, fontweight='bold',
                transform=ax.get_xaxis_transform())


def draw_simple_bars(ax, heights, face_colors=None, edge_colors=None, hatches=None):
    if face_colors is None:
        face_colors = bar_face_colors
    if edge_colors is None:
        edge_colors = bar_edge_colors
    if hatches is None:
        hatches = bar_hatches
    for x, h, fc, ec, ht in zip(bar_positions, heights,
                                face_colors, edge_colors, hatches):
        ax.bar(x, h, width=bar_width,
               color=fc, edgecolor=ec, hatch=ht, linewidth=1.0)


# ============================================================================
# DATA LOADING — COST
# ============================================================================

cost_ref_all      = np.zeros([number_runs, length_sim])
cost_cm_all       = np.zeros([number_runs, length_sim])
cost_CfD_all      = np.zeros([number_runs, length_sim])
cost_flex_all     = np.zeros([number_runs, length_sim])
cost_scarcity_all = np.zeros([number_runs, length_sim])
carbon_tax_all    = np.zeros([number_runs, length_sim])
cost_merchant_all = np.zeros([number_runs, length_sim])
cost_existing_all = np.zeros([number_runs, length_sim])

# prices_mean_all: mean normalised cost trajectory (cost/demand), used for
# stacked bar rescaling.  cost_per_seed: per-seed window mean of cost/demand,
# used for uncertainty error bars.
prices_mean_all = np.zeros([number_runs, length_sim])
cost_per_seed   = []

for i in range(number_runs):
    folder = f"{data_path}{checkpoint_runs[i]}"

    # Per-seed total system cost — shape (steps, n_seeds)
    c0_raw = np.array(pd.read_csv(
        os.path.join(folder, 'cost_spot_other_markets_0.csv')).iloc[:, 1:])

    # Average demand — shape (steps, 1) for broadcasting
    demand = np.array(pd.read_csv(
        os.path.join(folder, 'average_demand.csv')).iloc[:, 1:])[T_START:T_END, 0:1]

    # Demand-normalised cost per seed, mean over time window -> (n_seeds,)
    cost_per_seed.append(
        np.mean(c0_raw[T_START:T_END, :] / demand, axis=0))

    # Mean over seeds of demand-normalised cost -> trajectory for bar rescaling
    prices_mean_all[i, :] = np.mean(
        c0_raw[T_START:T_END, :] / demand, axis=1)

    cost_markets_temp = np.array(
        pd.read_csv(os.path.join(folder, 'cost_spot_other_markets.csv')).iloc[:, 1:])
    cost_ref_all[i, :]      = cost_markets_temp[T_START:T_END, 0]
    cost_cm_all[i, :]       = cost_markets_temp[T_START:T_END, 1]
    cost_CfD_all[i, :]      = cost_markets_temp[T_START:T_END, 2]
    cost_flex_all[i, :]     = cost_markets_temp[T_START:T_END, 3]
    cost_scarcity_all[i, :] = cost_markets_temp[T_START:T_END, 4]
    carbon_tax_all[i, :]    = cost_markets_temp[T_START:T_END, 5]
    cost_merchant_all[i, :] = cost_markets_temp[T_START:T_END, 6]
    cost_existing_all[i, :] = cost_markets_temp[T_START:T_END, 7]

mean_cm_net        = np.zeros(number_runs)
mean_CfD_net       = np.zeros(number_runs)
mean_flex_comp     = np.zeros(number_runs)
mean_scarcity_comp = np.zeros(number_runs)
mean_carbon_tax    = np.zeros(number_runs)
mean_merchant_comp = np.zeros(number_runs)
mean_existing_comp = np.zeros(number_runs)
mean_others_comp   = np.zeros(number_runs)

for k in range(number_runs):
    price_k = prices_mean_all[k, :]   # already demand-normalised [EUR/MWh]
    ref_k   = cost_ref_all[k, :]      # raw total cost (used as denominator ratio)

    cm_pu       = (cost_cm_all[k, :]       / ref_k) * price_k
    cfd_pu      = (cost_CfD_all[k, :]      / ref_k) * price_k
    flex_pu     = (cost_flex_all[k, :]     / ref_k) * price_k
    scarcity_pu = (cost_scarcity_all[k, :] / ref_k) * price_k
    tax_pu      = (carbon_tax_all[k, :]    / ref_k) * price_k
    merchant_pu = (cost_merchant_all[k, :] / ref_k) * price_k
    existing_pu = (cost_existing_all[k, :] / ref_k) * price_k

    mean_cm_net[k]        = np.mean(cm_pu)
    mean_CfD_net[k]       = np.mean(cfd_pu)
    mean_flex_comp[k]     = np.mean(flex_pu)
    mean_scarcity_comp[k] = np.mean(scarcity_pu)
    mean_carbon_tax[k]    = np.mean(tax_pu)
    mean_merchant_comp[k] = np.mean(merchant_pu)
    mean_existing_comp[k] = np.mean(existing_pu)

    named_pu = (cm_pu + cfd_pu + flex_pu + scarcity_pu
                + tax_pu + merchant_pu + existing_pu)
    mean_others_comp[k] = np.mean(price_k - named_pu)

cost_mean = np.array([np.mean(s)           for s in cost_per_seed])
cost_p25  = np.array([np.percentile(s, 25) for s in cost_per_seed])
cost_p75  = np.array([np.percentile(s, 75) for s in cost_per_seed])
cost_p5   = np.array([np.percentile(s,  5) for s in cost_per_seed])
cost_p95  = np.array([np.percentile(s, 95) for s in cost_per_seed])

# ============================================================================
# DATA LOADING — EMISSIONS
# ============================================================================

mean_emiss  = np.zeros(number_runs)
emiss_p25   = np.zeros(number_runs)
emiss_p75   = np.zeros(number_runs)
emiss_p5    = np.zeros(number_runs)
emiss_p95   = np.zeros(number_runs)

for i, folder_name in enumerate(checkpoint_runs):
    folder   = os.path.join(data_path, folder_name)
    emiss    = np.array(pd.read_csv(os.path.join(folder, 'CO2_emissions.csv')).iloc[:, 1:])
    per_seed = np.sum(emiss[T_START:T_END, :], axis=0) / (1e6 * 15)
    mean_emiss[i] = np.mean(per_seed)
    emiss_p25[i]  = np.percentile(per_seed, 25)
    emiss_p75[i]  = np.percentile(per_seed, 75)
    emiss_p5[i]   = np.percentile(per_seed,  5)
    emiss_p95[i]  = np.percentile(per_seed, 95)

# ============================================================================
# DATA LOADING — EENS
# ============================================================================

mean_eens = np.zeros(number_runs)
eens_p25  = np.zeros(number_runs)
eens_p75  = np.zeros(number_runs)
eens_p5   = np.zeros(number_runs)
eens_p95  = np.zeros(number_runs)

for i, folder_name in enumerate(checkpoint_runs):
    folder = os.path.join(data_path, folder_name)
    ens    = np.array(pd.read_csv(
        os.path.join(folder, 'energy_not_served.csv')).iloc[:, 1:])

    # Average demand — shape (steps, 1) for broadcasting across seeds
    demand = np.array(pd.read_csv(
        os.path.join(folder, 'average_demand.csv')).iloc[:, 1:])[T_START:T_END, 0:1]

    # ENS ratio per seed = mean_t( ens[t,s] * 61.2 / demand[t] ) * 100  [%]
    per_seed = np.mean(
        ens[T_START:T_END, :] * 61.2 / demand, axis=0) * 100

    mean_eens[i] = np.mean(per_seed)
    eens_p25[i]  = np.percentile(per_seed, 25)
    eens_p75[i]  = np.percentile(per_seed, 75)
    eens_p5[i]   = np.percentile(per_seed,  5)
    eens_p95[i]  = np.percentile(per_seed, 95)

# ============================================================================
# DATA LOADING — CAPACITY
# ============================================================================

capacity_2037_mean          = np.zeros([number_runs, n_tech])
capacity_2037_max           = np.zeros([number_runs, n_tech])
capacity_2037_min           = np.zeros([number_runs, n_tech])
capacity_2037_percentile_75 = np.zeros([number_runs, n_tech])
capacity_2037_percentile_25 = np.zeros([number_runs, n_tech])
capacity_merchant_2037      = np.zeros([number_runs, n_tech])
capacity_cm_2037            = np.zeros([number_runs, n_tech])
capacity_CfD_2037           = np.zeros([number_runs, n_tech])
capacity_flexibility_2037   = np.zeros([number_runs, n_tech])
capacity_existing_2037      = np.zeros([number_runs, n_tech])

for i in range(number_runs):
    folder = f"{data_path}{checkpoint_runs[i]}"

    cap_tmp  = np.array(pd.read_csv(os.path.join(folder, 'capacity_2040.csv')).iloc[:, 1:]) / 1e3
    me_tmp   = np.array(pd.read_csv(os.path.join(folder, 'capacity_2040_merchant.csv')).iloc[:, 1:]) / 1e3
    cm_tmp   = np.array(pd.read_csv(os.path.join(folder, 'capacity_2040_cm.csv')).iloc[:, 1:]) / 1e3
    cfd_tmp  = np.array(pd.read_csv(os.path.join(folder, 'capacity_2040_CfD.csv')).iloc[:, 1:]) / 1e3
    flex_tmp = np.array(pd.read_csv(os.path.join(folder, 'capacity_2040_flexibility.csv')).iloc[:, 1:]) / 1e3

    capacity_2037_mean[i, :]          = np.mean(cap_tmp,  axis=1)
    capacity_2037_max[i, :]           = np.percentile(cap_tmp, 95, axis=1)
    capacity_2037_min[i, :]           = np.percentile(cap_tmp,  5, axis=1)
    capacity_2037_percentile_75[i, :] = np.percentile(cap_tmp, 75, axis=1)
    capacity_2037_percentile_25[i, :] = np.percentile(cap_tmp, 25, axis=1)

    capacity_merchant_2037[i, :]    = np.mean(me_tmp,   axis=1)
    capacity_cm_2037[i, :]          = np.mean(cm_tmp,   axis=1)
    capacity_CfD_2037[i, :]         = np.mean(cfd_tmp,  axis=1)
    capacity_flexibility_2037[i, :] = np.mean(flex_tmp, axis=1)
    capacity_existing_2037[i, :]    = (capacity_2037_mean[i, :] - capacity_cm_2037[i, :]
                                       - capacity_merchant_2037[i, :] - capacity_CfD_2037[i, :]
                                       - capacity_flexibility_2037[i, :])

total_capacity_2037 = np.clip(capacity_2037_mean, 0, None)

# ============================================================================
# FIGURE LAYOUT
# ============================================================================
LEFT_W   = 1
RIGHT_W  = 2
COST_H   = 2
EMISS_H  = 1
EENS_H   = 1

fig = plt.figure(figsize=(14, 10))

outer_gs = gridspec.GridSpec(
    1, 2,
    figure=fig,
    width_ratios=[LEFT_W, RIGHT_W],
    wspace=0.15,
)

left_gs = gridspec.GridSpecFromSubplotSpec(
    3, 1,
    subplot_spec=outer_gs[0, 0],
    height_ratios=[COST_H, EMISS_H, EENS_H],
    hspace=0.3,
)
ax_cost  = fig.add_subplot(left_gs[0, 0])
ax_emiss = fig.add_subplot(left_gs[1, 0])
ax_eens  = fig.add_subplot(left_gs[2, 0])

right_gs = gridspec.GridSpecFromSubplotSpec(
    4, 2,
    subplot_spec=outer_gs[0, 1],
    hspace=0.25,
    wspace=0.15,
)
cap_axes = [[fig.add_subplot(right_gs[r, c]) for c in range(2)] for r in range(4)]

tech_layout = [
    (0, "Solar PV",         False, 0, 0),
    (1, "Onshore Wind",     False, 0, 1),
    (2, "Offshore Wind",    False, 0, 2),
    (6, "Batteries – 3 h",  True, 0, 3),
    (3, "Coal",             True, 1, 0),
    (4, "OCGT",             False, 1, 1),
    (5, "CCGT",             False, 1, 2),
    (7, "Batteries – 8 h",  True, 1, 3),
]

# ============================================================================
# HELPER — draw one capacity subplot
# ============================================================================

def draw_capacity_subplot(ax, int_graph, title, show_y_label,
                          is_last_row=False, show_legend=False):
    ex  = capacity_existing_2037[run_order, int_graph]
    me  = capacity_merchant_2037[run_order, int_graph]
    cm  = capacity_cm_2037[run_order, int_graph]
    cfd = capacity_CfD_2037[run_order, int_graph]
    fl  = capacity_flexibility_2037[run_order, int_graph]

    ax.bar(bar_positions, ex,  color=colors[0], edgecolor=colors[0], label="Existing")
    ax.bar(bar_positions, me,  bottom=ex,
           color=colors[2], edgecolor=colors[2], label="Merchant")
    ax.bar(bar_positions, cm,  bottom=ex + me,
           color='white', hatch='xx', edgecolor=colors[3], label="CM")
    ax.bar(bar_positions, cfd, bottom=ex + me + cm,
           color=colors[4], edgecolor=colors[4], label="CfD")
    ax.bar(bar_positions, fl, bottom=ex + me + cm + cfd,
           color='white', hatch='oo', edgecolor=colors[5], label="Flexibility")

    top = ex + me + cm + cfd + fl

    for mn, mx, capsize in [
        (capacity_2037_min[run_order, int_graph],
         capacity_2037_max[run_order, int_graph], 0),
        (capacity_2037_percentile_25[run_order, int_graph],
         capacity_2037_percentile_75[run_order, int_graph], 5),
    ]:
        err_lo = np.clip(total_capacity_2037[run_order, int_graph] - mn, 0, None)
        err_hi = np.clip(mx - total_capacity_2037[run_order, int_graph], 0, None)
        ax.errorbar(bar_positions, top, yerr=np.array([err_lo, err_hi]),
                    fmt='none', ecolor='black', capsize=capsize)

    for pos in separator_positions:
        ax.axvline(x=pos, color='grey', linestyle='--', linewidth=0.5)

    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_title(title, fontsize=10, fontweight='bold', pad=3)

    if show_y_label:
        ax.set_ylabel("[GW]", fontsize=9)

    if is_last_row:
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, ha='center', fontsize=9)
        draw_xaxis_layers(ax)
    else:
        ax.set_xticks([])

    _ = show_legend

# ============================================================================
# PLOT — TOTAL COST
# ============================================================================

cm_pos  = np.maximum(mean_cm_net,  0)
cm_neg  = np.minimum(mean_cm_net,  0)
cfd_pos = np.maximum(mean_CfD_net, 0)
cfd_neg = np.minimum(mean_CfD_net, 0)

color_existing = colors[0]
color_merchant = colors[2]
color_others   = '#7F7F7F'

ax_cost.bar(bar_positions, mean_scarcity_comp, width=bar_width,
            color=color_scarcity, edgecolor=color_scarcity, label='EEN')

b2 = mean_scarcity_comp
ax_cost.bar(bar_positions, cm_pos, width=bar_width, bottom=b2,
            color='white', hatch='xx', edgecolor=color_cm, label='CRM')

b3 = b2 + cm_pos
ax_cost.bar(bar_positions, cfd_pos, width=bar_width, bottom=b3,
            color=color_CfD, edgecolor=color_CfD, label='CfD')

b4 = b3 + cfd_pos
ax_cost.bar(bar_positions, mean_flex_comp, width=bar_width, bottom=b4,
            color='white', hatch='oo', edgecolor=color_flexibility, label='Flexibility')

b5 = b4 + mean_flex_comp

ax_cost.bar(bar_positions, mean_carbon_tax,  width=bar_width, bottom=b5,
           color=color_tax_return, edgecolor='black', hatch='////', label='ETS recovery')

b6 = b5 + mean_carbon_tax   

ax_cost.bar(bar_positions, mean_existing_comp, width=bar_width, bottom=b6,
            color=color_existing, edgecolor=color_existing, label='Existing')

b7 = b6 + mean_existing_comp
ax_cost.bar(bar_positions, mean_merchant_comp, width=bar_width, bottom=b7,
            color=color_merchant, edgecolor=color_merchant, label='Merchant')

b8 = b7 + mean_merchant_comp
others_pos = np.maximum(mean_others_comp, 0)
others_neg = np.minimum(mean_others_comp, 0)
ax_cost.bar(bar_positions, others_pos, width=bar_width, bottom=b8,
            color=color_others, edgecolor=color_others, label='Others')

ax_cost.bar(bar_positions, cfd_neg, width=bar_width,
            color=color_CfD, edgecolor=color_CfD, alpha=0.5)
ax_cost.bar(bar_positions, cm_neg, width=bar_width, bottom=cfd_neg,
            color='white', hatch='xx', edgecolor=color_cm, alpha=0.5)
ax_cost.bar(bar_positions, others_neg, width=bar_width, bottom=cfd_neg + cm_neg,
            color=color_others, edgecolor=color_others, alpha=0.5)

ax_cost.errorbar(bar_positions, cost_mean,
                 yerr=np.clip([cost_mean - cost_p5,  cost_p95 - cost_mean], 0, None),
                 fmt='none', ecolor='black', capsize=0, linewidth=0.8)
ax_cost.errorbar(bar_positions, cost_mean,
                 yerr=np.clip([cost_mean - cost_p25, cost_p75 - cost_mean], 0, None),
                 fmt='none', ecolor='black', capsize=5, linewidth=1.5)

ax_cost.axhline(y=0, color='black', linewidth=0.6)
for sp in separator_positions:
    ax_cost.axvline(x=sp, color='grey', linestyle='--', linewidth=0.5)

ax_cost.set_xticks(x_tick_positions)
ax_cost.set_xticklabels(x_tick_labels, ha='center', fontsize=10)
ax_cost.set_ylabel('[€/MWh]', fontsize=11)
ax_cost.grid(axis='y', alpha=0.3)
ax_cost.yaxis.set_major_locator(plt.MaxNLocator(4))
ax_cost.set_ylim(0, 100)
ax_cost.set_title("Total Unitary Cost", fontsize=12, fontweight='bold', pad=6)

# ============================================================================
# PLOT — EMISSIONS
# ============================================================================

draw_simple_bars(ax_emiss, mean_emiss)

ax_emiss.errorbar(bar_positions, mean_emiss,
                  yerr=np.clip([mean_emiss - emiss_p5,  emiss_p95 - mean_emiss], 0, None),
                  fmt='none', ecolor='black', capsize=0, linewidth=0.8)
ax_emiss.errorbar(bar_positions, mean_emiss,
                  yerr=np.clip([mean_emiss - emiss_p25, emiss_p75 - mean_emiss], 0, None),
                  fmt='none', ecolor='black', capsize=5, linewidth=1.5)

ax_emiss.axhline(y=0, color='black', linewidth=0.6)
for sp in separator_positions:
    ax_emiss.axvline(x=sp, color='grey', linestyle='--', linewidth=0.5)

ax_emiss.set_xticks(x_tick_positions)
ax_emiss.set_xticklabels(x_tick_labels, ha='center', fontsize=10)
ax_emiss.set_ylabel('[MtCO$_2$]', fontsize=11)
ax_emiss.grid(axis='y', alpha=0.3)
ax_emiss.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_emiss.set_title("Yearly Emissions", fontsize=12, fontweight='bold', pad=6)

# ============================================================================
# PLOT — EENS
# ============================================================================

draw_simple_bars(ax_eens, mean_eens,
                 face_colors=bar_face_colors_eens,
                 edge_colors=bar_edge_colors_eens,
                 hatches=bar_hatches_eens)

ax_eens.axhline(y=0, color='black', linewidth=0.6)
for sp in separator_positions:
    ax_eens.axvline(x=sp, color='grey', linestyle='--', linewidth=0.5)

ax_eens.set_xticks(x_tick_positions)
ax_eens.set_xticklabels(x_tick_labels, ha='center', fontsize=10)
draw_xaxis_layers(ax_eens)
ax_eens.set_ylabel('[%]', fontsize=11)
ax_eens.grid(axis='y', alpha=0.3)
ax_eens.yaxis.set_major_locator(plt.MaxNLocator(3))
ax_eens.set_title("EENS Ratio", fontsize=12, fontweight='bold', pad=6)

# ============================================================================
# PLOT — CAPACITY SUBPLOTS
# ============================================================================

for (int_graph, title, has_flex, cap_col, cap_row) in tech_layout:
    ax       = cap_axes[cap_row][cap_col]
    is_last  = (cap_row == 3)
    show_leg = (cap_row == 0 and cap_col == 1) or (cap_row == 2 and cap_col == 0)
    show_y_label = (cap_col == 0)
    draw_capacity_subplot(ax, int_graph, title, show_y_label,
                          is_last_row=is_last, show_legend=show_leg)

# ── Shared legend for the capacity column ─────────────────────────────────
legend_source_ax = cap_axes[3][0]
handles, labels = legend_source_ax.get_legend_handles_labels()

top_left_bbox  = cap_axes[0][0].get_position()
top_right_bbox = cap_axes[0][1].get_position()
cap_col_x_center = 0.5 * (top_left_bbox.x0 + top_right_bbox.x1)
cap_col_y_top    = top_left_bbox.y1

fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(cap_col_x_center, cap_col_y_top + 0.03),
    ncol=5,
    frameon=False,
    fontsize=10,
)

# ── Shared legend for the cost panel ──────────────────────────────────────
cost_handles, cost_labels = ax_cost.get_legend_handles_labels()
cost_bbox = ax_cost.get_position()
cost_x_center = 0.5 * (cost_bbox.x0 + cost_bbox.x1)
cost_y_top    = cost_bbox.y1

fig.legend(
    cost_handles, cost_labels,
    loc='lower center',
    bbox_to_anchor=(cost_x_center, cost_y_top + 0.03),
    ncol=4,
    frameon=False,
    fontsize=9,
    columnspacing=1.2,
    handlelength=1.5,
)

# ============================================================================
# SAVE
# ============================================================================

fig.savefig("plot_metrics_final_composite_V6.pdf", format="pdf", bbox_inches="tight")
print("Saved: plot_metrics_final_composite_V6.pdf")