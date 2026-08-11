"""
Synthetic data generator for illustrative percentile/threshold analysis.

Generates a fully synthetic usage dataset via percentile-anchored interpolation
(not a generic distribution with hopeful parameters), builds four illustrative
charts from it, and prints every underlying statistic to the console so it can
be checked against any accompanying written content -- every time this script
is run, not just the first time.

See README.md for the methodology write-up and the story of what went wrong
in earlier versions of this approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# Reproducibility is handled per-call via np.random.default_rng(seed) inside
# gen_month(). There is deliberately no global seed here: the modern Generator
# API does not read np.random.seed(), so setting it would imply a guarantee
# this script does not actually rely on.

OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

DARK = "#1D3557"
TEAL = "#2C7873"
LIGHT_TEAL = "#A8DADC"
MID_TEAL = "#52A0A0"
GRID = "#E5E5E5"
BG = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": "#CCCCCC", "axes.labelcolor": DARK, "text.color": DARK,
    "xtick.color": DARK, "ytick.color": DARK, "axes.titlecolor": DARK,
    "figure.facecolor": BG, "axes.facecolor": BG,
})

N = 6000
PREMIUM_SHARE = 0.10          # exactly 10% of the population is the "premium" tail
PREMIUM_VOLUME_TARGET = 0.48  # that 10% must account for 48% of total volume, by construction
FREE_CAP = 8
MID_CAP = 20


def gen_month(seed_offset=0, p90_target=None):
    """
    Generate one month of synthetic per-user usage counts.

    The bulk of the population (90%) is built via percentile-anchored
    interpolation, so its percentiles land close to the intended targets
    by construction rather than by chance. The remaining 10% is a deliberately
    heavy-tailed "premium" segment, rescaled so its share of total volume
    hits PREMIUM_VOLUME_TARGET exactly.

    p90_target lets the casual population's ceiling vary slightly per call --
    used to give a monitoring-window chart organic month-to-month movement
    instead of an artificially flat line.
    """
    rng = np.random.default_rng(100 + seed_offset)
    n_casual = int(N * (1 - PREMIUM_SHARE))
    n_premium = N - n_casual

    p90_val = MID_CAP if p90_target is None else p90_target
    anchor_pct = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
    anchor_val = [0,   0,    1,    3,    FREE_CAP, p90_val]
    # Divide by N, not n_casual: the casual segment occupies percentiles 0-90 of
    # the FULL population, so its ranks must be expressed against the full N.
    # Dividing by n_casual would stretch it across 0-100 and break the anchors.
    ranks = (np.arange(n_casual) + 0.5) / N
    target_curve = np.interp(ranks, anchor_pct, anchor_val)
    casual_vals = rng.poisson(np.clip(target_curve, 0.05, None))
    casual_vals = np.clip(casual_vals, 0, p90_val)

    casual_sum = casual_vals.sum()
    target_premium_sum = PREMIUM_VOLUME_TARGET * casual_sum / (1 - PREMIUM_VOLUME_TARGET)
    tail_shape = rng.pareto(a=2.2, size=n_premium) + 1
    premium_vals = tail_shape / tail_shape.sum() * target_premium_sum
    premium_vals = np.round(premium_vals).astype(int)
    premium_vals = np.clip(premium_vals, p90_val + 1, None)

    return np.concatenate([casual_vals, premium_vals])


def add_caption(fig, text):
    fig.text(0.01, 0.975, text, fontsize=10.5, style='italic', color="#555555", va='top')


def main():
    single_month = gen_month(seed_offset=0)
    p25, p50, p75, p90, p95, p99 = [np.percentile(single_month, p) for p in [25, 50, 75, 90, 95, 99]]
    free_cov = (single_month <= FREE_CAP).mean() * 100
    mid_cov = (single_month <= MID_CAP).mean() * 100
    prem_mask = single_month > MID_CAP
    prem_user_pct = prem_mask.mean() * 100
    prem_vol_pct = single_month[prem_mask].sum() / single_month.sum() * 100

    print("=== COMPUTED STATISTICS -- verify all prose against these before publishing ===")
    print(f"P25={p25:.0f} P50(median)={p50:.0f} P75={p75:.0f} P90={p90:.0f} P95={p95:.0f} P99={p99:.0f}")
    print(f"Free Cap ({FREE_CAP}) coverage = {free_cov:.0f}%")
    print(f"Mid Cap ({MID_CAP}) coverage = {mid_cov:.0f}%")
    print(f"Premium users = {prem_user_pct:.0f}% of base, drive {prem_vol_pct:.1f}% of volume")
    # The rescale in gen_month() hits PREMIUM_VOLUME_TARGET exactly, but the
    # np.round() and the floor clip that follow both move the sum. Print target
    # against achieved so that drift is visible rather than assumed away.
    print(f"  -> target was {PREMIUM_VOLUME_TARGET * 100:.0f}%; "
          f"drift of {prem_vol_pct - PREMIUM_VOLUME_TARGET * 100:+.1f}pp from rounding and the floor clip")

    # ---- Chart 1: monthly stability check ----
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_data = [gen_month(seed_offset=i) for i in range(12)]
    q1s = [np.percentile(m, 25) for m in monthly_data]
    q3s = [np.percentile(m, 75) for m in monthly_data]
    meds = [np.percentile(m, 50) for m in monthly_data]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.boxplot(monthly_data, tick_labels=months, showfliers=True, patch_artist=True,
               medianprops=dict(color=DARK, linewidth=2),
               boxprops=dict(facecolor=LIGHT_TEAL, edgecolor=TEAL),
               whiskerprops=dict(color=TEAL), capprops=dict(color=TEAL),
               flierprops=dict(marker='o', markersize=2, markerfacecolor=MID_TEAL, markeredgecolor='none', alpha=0.4))
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylim(0, 120)
    tick_vals = [0, 1, 3, 10, 30, 100]
    ax.set_yticks(tick_vals)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylabel("Value per user / month")
    ax.set_title("Monthly Distribution Stability -- 12 Months", fontsize=14, fontweight='bold', loc='left', pad=14)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    add_caption(fig, f"Median stays at {int(np.median(meds))}, IQR holds {int(np.median(q1s))}-{int(np.median(q3s))} across every month")
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig(os.path.join(OUT_DIR, "chart1_stability.png"), dpi=200)
    plt.close()

    # ---- Chart 2: distribution shape + cumulative coverage ----
    max_x = 60
    bins = np.arange(0, max_x + 2) - 0.5
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax1 = axes[0]
    counts_hist, edges, patches = ax1.hist(single_month[single_month <= max_x], bins=bins,
                                            color=LIGHT_TEAL, edgecolor=TEAL, linewidth=0.5)
    for i, patch in enumerate(patches):
        center = (edges[i] + edges[i+1]) / 2
        patch.set_facecolor(LIGHT_TEAL if center <= FREE_CAP else (MID_TEAL if center <= MID_CAP else DARK))
    ax1.axvline(FREE_CAP, color=TEAL, linestyle='--', linewidth=1.5)
    ax1.axvline(MID_CAP, color=DARK, linestyle='--', linewidth=1.5)
    ax1.set_xlabel("Value per user / month")
    ax1.set_ylabel("Number of users")
    ax1.set_title("Distribution Shape", fontsize=13, fontweight='bold', loc='left', pad=12)
    ax1.grid(axis='y', color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    for s in ['top', 'right']:
        ax1.spines[s].set_visible(False)

    ax2 = axes[1]
    sorted_counts = np.sort(single_month)
    cum_pct = np.arange(1, len(sorted_counts)+1) / len(sorted_counts) * 100
    ax2.plot(sorted_counts, cum_pct, color=TEAL, linewidth=2.5)
    ax2.set_xlim(0, max_x)
    ax2.set_ylim(0, 102)
    ax2.axvline(FREE_CAP, color=TEAL, linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(MID_CAP, color=DARK, linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel("Value per user / month")
    ax2.set_ylabel("Cumulative % of users")
    ax2.set_title("Cumulative Coverage", fontsize=13, fontweight='bold', loc='left', pad=12)
    ax2.grid(axis='y', color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    for s in ['top', 'right']:
        ax2.spines[s].set_visible(False)
    add_caption(fig, f"~{free_cov:.0f}% of users never exceed {FREE_CAP}; the curve flattens sharply after {MID_CAP}")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(OUT_DIR, "chart2_distribution.png"), dpi=200)
    plt.close()

    # ---- Chart 3: segment concentration ----
    free_mask = single_month <= FREE_CAP
    mid_mask = (single_month > FREE_CAP) & (single_month <= MID_CAP)
    tier_names = ["Segment A", "Segment B", "Segment C"]
    user_counts_tier = [free_mask.sum(), mid_mask.sum(), prem_mask.sum()]
    volume_tier = [single_month[free_mask].sum(), single_month[mid_mask].sum(), single_month[prem_mask].sum()]
    tier_colors = [LIGHT_TEAL, MID_TEAL, DARK]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax1 = axes[0]
    ax1.bar(tier_names, user_counts_tier, color=tier_colors, width=0.55)
    ax1.set_ylim(0, max(user_counts_tier) * 1.15)
    ax1.set_ylabel("Number of users")
    ax1.set_title("User Share by Segment", fontsize=13, fontweight='bold', loc='left', pad=12)
    ax1.grid(axis='y', color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    for s in ['top', 'right']:
        ax1.spines[s].set_visible(False)

    ax2 = axes[1]
    ax2.bar(tier_names, volume_tier, color=tier_colors, width=0.55)
    ax2.set_ylim(0, max(volume_tier) * 1.18)
    ax2.set_ylabel("Total volume")
    ax2.set_title("Volume Share by Segment", fontsize=13, fontweight='bold', loc='left', pad=12)
    ax2.grid(axis='y', color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    for s in ['top', 'right']:
        ax2.spines[s].set_visible(False)
    add_caption(fig, f"Segment C is ~{prem_user_pct:.0f}% of users but drives ~{prem_vol_pct:.0f}% of volume")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(OUT_DIR, "chart3_concentration.png"), dpi=200)
    plt.close()

    # ---- Chart 4: monitoring-window stress test (deliberately NOT perfectly flat) ----
    monitor_months = ["Jul","Aug","Sep","Oct","Nov","Dec"]
    monitor_rng = np.random.default_rng(55)
    p90_targets = monitor_rng.integers(16, MID_CAP + 1, size=6)
    monitor_data = [gen_month(seed_offset=200+i, p90_target=int(p90_targets[i])) for i in range(6)]
    p90_series = [np.percentile(m, 90) for m in monitor_data]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.plot(monitor_months, p90_series, color=DARK, linewidth=2.2, marker='o', markersize=4)
    ax.axhline(MID_CAP, color="#999999", linestyle='--', linewidth=1)
    ax.set_ylim(0, MID_CAP + 6)
    ax.set_ylabel("90th percentile value")
    ax.set_title("Monitoring Window Stress Test", fontsize=14, fontweight='bold', loc='left', pad=14)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    p90_min, p90_max = min(p90_series), max(p90_series)
    add_caption(fig, f"90th percentile moved between {p90_min:.0f} and {p90_max:.0f} -- never exceeding the threshold of {MID_CAP}")
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig(os.path.join(OUT_DIR, "chart4_stress_test.png"), dpi=200)
    plt.close()

    print(f"\nP90 monitoring series: {[f'{v:.0f}' for v in p90_series]}")
    print(f"\nAll charts saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
