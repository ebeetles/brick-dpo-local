"""Figures for the midway report (SPEC §7.3), written to report/figures/*.pdf.

Real points/distributions come only from reportable runs in runs/aggregate.json. Models without runs appear
as hollow legend entries labeled "pending" with no position. Single column: 3.3 in wide, 8 pt text.
Each model has a fixed color and marker (midway/config.py MODELS) shared by all figures.

Run:  make report
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from midway.aggregate import load_aggregate, load_records, primary_run_set  # noqa: E402
from midway.config import FIGURES_DIR, GROUND_TRUTH_COLOR, MODELS, REPO_ROOT, SETTINGS  # noqa: E402

WIDTH = 3.3
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e4e4e0', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'lines.linewidth': 2, 'lines.markersize': 6,
})
INK_MUTED = '#6b6b6b'
PENDING_MODELS = ['RFT', 'DPO-seq', 'DPO-prefix', 'DPO-local']


def pending_handle(ax, model: str):
    """Legend-only hollow marker for a model that has no runs yet (plotted at no position)."""
    style = MODELS[model]
    ax.scatter([], [], marker=style['marker'], facecolors='none', edgecolors=style['color'], linewidths=1.2,
               label=f'{model} (pending)')


def ci_err(ci: dict, scale: float = 1) -> list[list[float]]:
    """Asymmetric error bar [[mean - lo], [hi - mean]] for matplotlib."""
    return [[scale * (ci['mean'] - ci['lo'])], [scale * (ci['hi'] - ci['mean'])]]


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f'{name}.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}')


def fig2_stability_vs_cost(agg: dict) -> None:
    """x = median seconds per build, y = final % stable (rollback on), one point per model with CI bars."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    full = primary_run_set(agg, 'full')
    stable = full and full['metrics']['stable']
    sec = full and full['metrics']['sec_per_build_median']
    if stable and sec and stable['mean'] is not None and sec['mean'] is not None:
        style = MODELS['BrickGPT']
        ax.errorbar(sec['mean'], 100 * stable['mean'], xerr=ci_err(sec), yerr=ci_err(stable, 100),
                    fmt=style['marker'], color=style['color'], ecolor=style['color'], elinewidth=1, capsize=2,
                    label=f'BrickGPT ({full["prompts"]})')
        ax.set_xlim(left=0)
    else:
        pending_handle(ax, 'BrickGPT')
        ax.text(0.5, 0.92, 'no runs yet', transform=ax.transAxes, ha='center', va='center', color=INK_MUTED)
        ax.set_xticks([])
    for model in PENDING_MODELS:
        pending_handle(ax, model)
    ax.set_ylim(0, 102)
    ax.set_xlabel('Median seconds per build')
    ax.set_ylabel(r'Final % stable (rollback on)')
    ax.legend(loc='lower right', frameon=False, handletextpad=0.3, borderaxespad=0.2)
    save(fig, 'fig2_stability_vs_cost')


def fig3_brick_counts(agg: dict) -> None:
    """One horizontal box per model/setting: distribution of bricks per build. Pending slots are labeled."""
    slots = [('Ground truth', 'gt', GROUND_TRUTH_COLOR)]
    slots += [(f'BrickGPT, {SETTINGS[s]["label"]}', s, MODELS['BrickGPT']['color']) for s in ('norsnorb', 'rs', 'full')]
    slots += [(model, None, MODELS[model]['color']) for model in PENDING_MODELS]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    labels = []
    for pos, (label, setting, color) in enumerate(reversed(slots)):
        run_set = primary_run_set(agg, setting) if setting else None
        counts = [r['n_bricks'] for d in run_set['runs'] for r in load_records(REPO_ROOT / d)
                  if r['n_bricks'] is not None] if run_set else []
        if counts:
            ax.boxplot(counts, positions=[pos], vert=False, widths=0.5, patch_artist=True,
                       boxprops={'facecolor': color, 'edgecolor': color, 'alpha': 0.35},
                       medianprops={'color': color, 'linewidth': 2},
                       whiskerprops={'color': color}, capprops={'color': color},
                       flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': color,
                                   'markeredgecolor': 'none', 'alpha': 0.6})
            labels.append(f'{label} (n={len(counts)})')
        else:
            ax.text(0.02, pos, 'pending', transform=ax.get_yaxis_transform(), va='center', color=INK_MUTED)
            labels.append(label)
    ax.set_yticks(range(len(slots)), labels)
    ax.set_ylim(-0.7, len(slots) - 0.3)
    ax.set_xlim(left=0)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Bricks per build')
    save(fig, 'fig3_brick_counts')


def fig4_stable_vs_pairs(agg: dict) -> None:
    """x = preference-pair budget, y = Stable@1. DPO lines pending; BrickGPT as a reference line if it exists."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    budgets = [500, 2000, 5000]
    rs = primary_run_set(agg, 'rs')
    stable = rs and rs['metrics']['stable']
    if stable and stable['mean'] is not None:
        color = MODELS['BrickGPT']['color']
        ax.axhspan(100 * stable['lo'], 100 * stable['hi'], color=color, alpha=0.12, linewidth=0)
        ax.axhline(100 * stable['mean'], color=color, linewidth=1.5, linestyle='--',
                   label=f'BrickGPT Stable@1 ({rs["prompts"]})')
    else:
        ax.plot([], [], color=MODELS['BrickGPT']['color'], linestyle='--', linewidth=1.5,
                label='BrickGPT Stable@1 (pending)')
    for model in ['DPO-seq', 'DPO-prefix', 'DPO-local']:
        pending_handle(ax, model)
    ax.set_xscale('log')
    ax.set_xticks(budgets, ['500', '2k', '5k'])
    ax.minorticks_off()
    ax.set_xlim(350, 7000)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Preference pairs')
    ax.set_ylabel('Stable@1 (%)')
    ax.legend(loc='upper left', frameon=False, handletextpad=0.3, borderaxespad=0.2)
    save(fig, 'fig4_stable_vs_pairs')


def main() -> None:
    agg = load_aggregate()
    fig2_stability_vs_cost(agg)
    fig3_brick_counts(agg)
    fig4_stable_vs_pairs(agg)


if __name__ == '__main__':
    main()
