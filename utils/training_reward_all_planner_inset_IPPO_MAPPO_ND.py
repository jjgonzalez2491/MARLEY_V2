import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import pandas as pd
import os

matplotlib.rcParams['font.family'] = 'Nimbus Sans'
plt.rcParams.update({'font.size': 10})
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = '250526_CM_CfD_runs'

SEEDS = {
    'Seed 1': [11, 80],   # [IPPO key, MAPPO key]
    'Seed 2': [15, 81],
    'Seed 3': [59, 82],
}

SCENARIOS = ['IPPO', 'MAPPO']

METRICS = [
    ('planner',   'Planner Reward',   '[$]', None),
    ('incumbent', 'Incumbent Reward', '[$]', None),
    ('entrant',   'Entrant Reward',   '[$]', None),
]

INCUMBENT_COLS = [f'Agent_g_{i}' for i in range(9)]
ENTRANT_COLS   = [f'Agent_g_{i}' for i in range(9, 16)]
PLANNER_COL    = 'Agent_p_0'

MD_NAMES = list(SEEDS.keys())

COLORS = {
    'Seed 1': '#156082',
    'Seed 2': '#00E5FF',
    'Seed 3': '#7F7F7F',
}

LAST_PCT = 0.40

LW_MAIN  = 0.8   # linewidth for main plots
LW_INSET = 0.6   # linewidth for inset plots

# Inset position per row [left, bottom, width, height] in axes-fraction coords
INSET_RECT = [
    [0.25, 0.50, 0.7, 0.45],   # row 0 — upper right
    [0.25, 0.15, 0.7, 0.45],   # row 1 — lower right
    [0.25, 0.15, 0.7, 0.45],   # row 2 — lower right
]

# Set to None to autoscale, or (ymin, ymax) to fix manually
ROW_YLIMS = [
    (-0.34, -0.1),    # row 0 — Planner Reward
    (-6, 1.5),       # row 1 — Incumbent Reward
    (-5, 0.1),     # row 2 — Entrant Reward
]

INSET_YLIMS = [
    (-0.325, -0.3),    # row 0 — Planner Reward
    (1.2, 1.4),       # row 1 — Incumbent Reward
    (0.01, 0.06),     # row 2 — Entrant Reward
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(prefix, key):
    path = os.path.join(DATA_DIR, f'{prefix}_{key}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def extract_metrics(key):
    am = load_csv('agent_mean', key)
    if am is None:
        return None
    max_steps = am['timesteps'].max()
    if max_steps == 0:
        return None
    x = am['timesteps'].values / max_steps

    planner = am[PLANNER_COL].values if PLANNER_COL in am.columns \
              else np.full(len(x), np.nan)

    inc_cols = [c for c in INCUMBENT_COLS if c in am.columns]
    incumbent = am[inc_cols].sum(axis=1).values if inc_cols \
                else np.full(len(x), np.nan)

    ent_cols = [c for c in ENTRANT_COLS if c in am.columns]
    entrant = am[ent_cols].sum(axis=1).values if ent_cols \
              else np.full(len(x), np.nan)

    return dict(x=x[5:], planner=planner[5:], incumbent=incumbent[5:], entrant=entrant[5:])


def plot_lines(ax, ci, metric_key, data_cache, last_pct_only=False, lw=1.0):
    for md_name, keys in SEEDS.items():
        key  = keys[ci]
        data = data_cache.get(key)
        if data is None:
            continue
        x = data['x']
        y = data[metric_key]
        if last_pct_only:
            mask = x >= (1.0 - LAST_PCT)
            x, y = x[mask], y[mask]
            if len(x) == 0:
                continue
        ax.plot(x, y, color=COLORS[md_name], linewidth=lw, alpha=0.9)


def add_connectors(ax, inset, cutoff, inset_rect):
    ymin, ymax = ax.get_ylim()
    inset_top_y = inset_rect[1] + inset_rect[3]
    corners = [
        ((cutoff, ymax), (inset_rect[0],                  inset_top_y)),
        ((1.0,    ymax), (inset_rect[0] + inset_rect[2],  inset_top_y)),
    ]
    for (xA, yA), (xB, yB) in corners:
        con = mpatches.ConnectionPatch(
            xyA=(xA, yA), xyB=(xB, yB),
            coordsA='data', coordsB='axes fraction',
            axesA=ax, axesB=ax,
            color='#888888', linewidth=0.6,
            linestyle=(0, (4, 3)),
            clip_on=False, zorder=5,
        )
        ax.add_artist(con)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(output_path='training_inset.pdf'):
    n_rows = len(METRICS)
    n_cols = len(SCENARIOS)

    all_keys = {key for keys in SEEDS.values() for key in keys}
    data_cache = {key: extract_metrics(key) for key in all_keys}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharey=False)
    fig.subplots_adjust(hspace=0.15, wspace=0.06,
                        left=0.12, right=0.99, top=0.95, bottom=0.14)

    inset_store = {}

    for ri, (metric_key, metric_label, ylabel_unit, _) in enumerate(METRICS):
        for ci, scenario in enumerate(SCENARIOS):
            ax = axes[ri, ci]

            plot_lines(ax, ci, metric_key, data_cache, last_pct_only=False, lw=LW_MAIN)

            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(labelsize=8)
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
            ax.set_axisbelow(True)

            if ri == 0:
                ax.set_title(scenario, fontweight='bold', pad=4)

            if ci == 0:
                ax.set_ylabel(f"{metric_label}")
                ax.yaxis.set_label_coords(-0.1, 0.5)
            else:
                ax.yaxis.set_tick_params(labelleft=False)

            if ri != n_rows - 1:
                ax.xaxis.set_tick_params(labelbottom=False)

            cutoff = 1.0 - LAST_PCT
            ax.axvspan(cutoff, 1.0, color='#CCCCCC', alpha=0.20, zorder=1)
            ax.axvline(cutoff, color='#888888', linewidth=0.8,
                       linestyle='--', zorder=2)

            inset = ax.inset_axes(INSET_RECT[ri])
            inset_store[(ri, ci)] = inset

            plot_lines(inset, ci, metric_key, data_cache,
                       last_pct_only=True, lw=LW_INSET)

            inset.xaxis.set_major_locator(plt.MaxNLocator(3))
            inset.yaxis.set_major_locator(plt.MaxNLocator(2))
            inset.tick_params(labelsize=6)
            inset.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)
            inset.set_axisbelow(True)

            for spine in inset.spines.values():
                spine.set_linewidth(0.7)
                spine.set_edgecolor('#555555')
            inset.spines['top'].set_visible(True)
            inset.spines['right'].set_visible(True)

    # ── Sync y-limits row by row ──────────────────────────────────────────────
    for ri in range(n_rows):
        row_axes = [axes[ri, ci] for ci in range(n_cols)]
        if ROW_YLIMS[ri] is not None:
            ymin, ymax = ROW_YLIMS[ri]
        else:
            ymin = min(ax.get_ylim()[0] for ax in row_axes)
            ymax = max(ax.get_ylim()[1] for ax in row_axes)
        for ax in row_axes:
            ax.set_ylim(ymin, ymax)

    # ── Sync y-limits insets ──────────────────────────────────────────────

    for ri in range(n_rows):
        row_insets = [inset_store[(ri, ci)] for ci in range(n_cols)]
        if INSET_YLIMS[ri] is not None:
            ymin, ymax = INSET_YLIMS[ri]
        else:
            ymin = min(inset.get_ylim()[0] for inset in row_insets)
            ymax = max(inset.get_ylim()[1] for inset in row_insets)
        for inset in row_insets:
            inset.set_ylim(ymin, ymax)

    # ── Connectors after ylim sync ────────────────────────────────────────────
    cutoff = 1.0 - LAST_PCT
    for ri in range(n_rows):
        for ci in range(n_cols):
            add_connectors(axes[ri, ci], inset_store[(ri, ci)],
                           cutoff, INSET_RECT[ri])

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [plt.Line2D([0], [0], color=COLORS[md], linewidth=2, label=md)
               for md in MD_NAMES]
    fig.legend(handles=handles, loc='lower center', ncol=len(MD_NAMES),
               frameon=False, bbox_to_anchor=(0.55, 0.04))

    fig.text(0.55, 0.09, 'Normalised Training Steps',
             ha='center', va='bottom')

    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {output_path}')


if __name__ == '__main__':
    make_figure(output_path='training_reward_all_planner_inset_IPPO_MAPPO_ND.pdf')
    print('Done.')